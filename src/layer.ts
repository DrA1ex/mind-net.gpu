import {ILayer} from "mind-net.js/engine/base";
import {Matrix1D, Matrix2D} from "mind-net.js/engine/matrix";
import {GPU, Kernel, Texture} from "gpu.js";

import * as GpuUtils from "./utils/gpu";

export class GpuWrappedLayer {
    get size() {return this.layer.size}
    get prevSize() {return this.layer.prevSize}
    get activation() {return this.layer.activation}

    get biases() {return this.layer.biases;}
    get weights() {return this.layer.weights;}

    input!: Matrix2D;
    prime!: Matrix2D;
    output!: Matrix2D;

    forwardKernel?: Kernel;
    backwardKernel?: Kernel;

    private acquiredKernels: Kernel[] = [];

    constructor(gpu: GPU, public readonly layer: ILayer, index: number, batchSize: number) {
        if (layer.index === 0) throw new Error("Input layer should not be wrapped");

        let res = GpuUtils.createForwardKernel(gpu, this, batchSize);
        this.forwardKernel = res[0];
        this.acquiredKernels.push(...res[1]);

        res = GpuUtils.createBackwardKernel(gpu, this, batchSize, index === 1);
        this.backwardKernel = res[0];
        this.acquiredKernels.push(...res[1]);
    }

    forward(input: Matrix2D, actualSize: number) {
        if (!this.forwardKernel) throw new Error("Kernel was destroyed");
        const res = (this.forwardKernel as any)(input, this.weights, this.biases, actualSize);
        this.input = input;
        this.prime = res.prime;
        this.output = res.result;

        return this.output;
    }

    backward(error: Matrix2D, actualSize: number) {
        if (!this.backwardKernel) throw new Error("Kernel was destroyed");

        const result = (this.backwardKernel as any)(error, this.prime, this.input, this.weights, actualSize) as any;

        return {
            dB: this._toArray(result.dB) as Matrix1D,
            dW: this._toArray(result.dW) as Matrix2D,
            dError: this._toArray(result.dError) as Matrix2D
        };
    }

    private _toArray(texture?: Texture | Float32Array) {
        if (texture && "toArray" in texture) {
            return texture.toArray();
        }

        return texture;
    }

    destroy() {
        this.forwardKernel = undefined;
        this.backwardKernel = undefined;

        for (let kernel of this.acquiredKernels) {
            kernel.destroy();
        }

        this.acquiredKernels = [];
    }
}
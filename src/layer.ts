import {ILayer} from "mind-net.js/engine/base";
import {Dropout, Matrix} from "mind-net.js";
import {GPU, Kernel} from "gpu.js";

import {GpuArray1D, GpuArray2D} from "./base";
import * as GpuUtils from "./utils/gpu";
import {BackwardKernelFn, ForwardKernelFn, KernelReturnT} from "./utils/gpu";

export class GpuWrappedLayer {
    get size() {return this.layer.size}
    get prevSize() {return this.layer.prevSize}
    get activation() {return this.layer.activation}

    get biases() {return this.layer.biases;}
    get weights() {return this.layer.weights;}

    input!: GpuArray2D;
    prime!: GpuArray2D;
    output!: GpuArray2D;

    forwardKernel?: ForwardKernelFn;
    backwardKernel?: BackwardKernelFn;

    private acquiredKernels: Kernel[] = [];
    private dropouts: Dropout[];

    constructor(gpu: GPU, public readonly layer: ILayer, index: number, batchSize: number) {
        if (layer.index === 0) throw new Error("Input layer should not be wrapped");

        const forwardKernelRes = GpuUtils.createForwardKernel(gpu, this, batchSize);
        this.forwardKernel = forwardKernelRes[0];
        this.acquiredKernels.push(...forwardKernelRes[1]);

        const backwardKernelRes = GpuUtils.createBackwardKernel(gpu, this, batchSize, index === 1);
        this.backwardKernel = backwardKernelRes[0];
        this.acquiredKernels.push(...backwardKernelRes[1]);

        this.dropouts = Matrix.fill(() => new Dropout(layer), batchSize);
    }

    forward(input: GpuArray2D, actualSize: number, isTraining: boolean) {
        if (!this.forwardKernel) throw new Error("Kernel was destroyed");
        const res = (this.forwardKernel)(input, this.weights, this.biases, actualSize);
        this.input = input;
        this.prime = res.prime;
        this.output = res.result;

        if (isTraining && this.layer.dropout > 0) {
            for (let i = 0; i < actualSize; i++) {
                this.dropouts[i].calculateMask();
                this.dropouts[i].applyMask(this.output[i]);
            }
        }

        return this.output;
    }

    backward(error: GpuArray2D, actualSize: number) {
        if (!this.backwardKernel) throw new Error("Kernel was destroyed");

        if (this.layer.dropout > 0) {
            for (let i = 0; i < actualSize; i++) {
                this.dropouts[i].applyMask(error[i]);
            }
        }

        const result = (this.backwardKernel)(error, this.prime, this.input, this.weights, actualSize);

        return {
            dB: this._toArray(result.dB) as GpuArray1D,
            dW: this._toArray(result.dW) as GpuArray2D,
            dError: this._toArray(result.dError) as GpuArray2D
        };
    }

    private _toArray(texture?: KernelReturnT): any {
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
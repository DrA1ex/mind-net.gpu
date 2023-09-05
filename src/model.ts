import {IModel} from "mind-net.js/engine/base";
import {Matrix1D, Matrix2D} from "mind-net.js/engine/matrix";
import {Iter} from "mind-net.js";

import {GPU, IGPUSettings} from "gpu.js";
import {GpuWrappedLayer} from "./layer";

export type GpuWrapperOptionsT = {
    batchSize: number,
    gpu?: IGPUSettings
}

export const GpuWrapperOptsDefault: GpuWrapperOptionsT = {
    batchSize: 128
}

export class GpuModelWrapper {
    private readonly gpu: GPU;
    private actualSize: number = 0;
    private _destroyed = false;

    public get isDestroyed() {return this._destroyed;}
    public readonly layers: GpuWrappedLayer[];
    public readonly batchSize: number;

    constructor(public readonly model: IModel, options: Partial<GpuWrapperOptionsT> = {}) {
        if (!model.isCompiled) throw new Error("Model should be compiled");

        const opts = {...GpuWrapperOptsDefault, ...options};

        this.gpu = new GPU(opts.gpu);
        this.batchSize = opts.batchSize;

        this.layers = new Array(model.layers.length - 1);
        for (let i = 1; i < model.layers.length; i++) {
            this.layers[i - 1] = new GpuWrappedLayer(this.gpu, model.layers[i], this.batchSize);
        }
    }

    public compute(input: Matrix2D) {
        this._assertNotDestroyed();

        const result: Matrix2D = [];
        for (const batch of Iter.partition(input, this.batchSize)) {
            const out = this.forward(batch);
            result.push(...out);
        }

        return result;
    }

    public train(input: Matrix2D, expected: Matrix2D) {
        this._assertNotDestroyed();

        this.model.beforeTrain();

        const shuffled = Iter.shuffled(Array.from(Iter.zip(input, expected)));
        for (const batch of Iter.partition(shuffled, this.batchSize)) {
            this.trainBatch(batch);
        }

        this.model.afterTrain();
    }

    public trainBatch(batch: [Matrix1D, Matrix1D][]) {
        this._assertNotDestroyed();

        const input = batch.map(b => b[0]);
        const expected = batch.map(b => b[1]);

        const predicted = this.forward(input);
        this.backward(predicted, expected);
    }

    destroy() {
        if (this.isDestroyed) return;

        for (const layer of this.layers) {
            layer.destroy();
        }

        this._destroyed = true;
    }

    private forward(input: Matrix2D): Matrix2D {
        if (input.length > this.batchSize) throw new Error("Input can't be greater than batchSize")

        this.actualSize = input.length;
        let nexInput = input;
        for (const layer of this.layers) {
            nexInput = layer.forward(nexInput, this.actualSize);
        }

        return nexInput.slice(0, this.actualSize);
    }

    private backward(input: Matrix2D, expected: Matrix2D) {
        if (!this.actualSize) throw new Error("Not ready for backward pass");
        if (input.length > this.batchSize || expected.length > this.batchSize) throw new Error("Input/Output can't be greater than batchSize")
        if (input.length !== expected.length) throw new Error("Input and expected data sizes doesn't match!");

        let nextError = input.map((p, i) => this.model.loss.calculateError(p, expected[i]));
        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];
            const res = layer.backward(nextError, this.actualSize);

            this.model.optimizer.updateWeights(layer.layer, res.dW, res.dB, this.model.epoch, 1);
            nextError = res.dError;
        }

        this.actualSize = 0;
    }

    private _assertNotDestroyed() {
        if (this.isDestroyed) throw new Error("Wrapped destroyed");
    }
}
import {IModel} from "mind-net.js/engine/base";
import {Matrix1D, Matrix2D} from "mind-net.js/engine/matrix";
import {Iter, Matrix} from "mind-net.js";

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
    private _lossCache: Matrix1D[];

    public get isDestroyed() {return this._destroyed;}
    public readonly layers: GpuWrappedLayer[];
    public readonly batchSize: number;

    public get inputSize() {return this.model.inputSize;}
    public get outputSize() {return this.model.outputSize;}

    constructor(public readonly model: IModel, options: Partial<GpuWrapperOptionsT> = {}) {
        if (!model.isCompiled) throw new Error("Model should be compiled");

        const opts = {...GpuWrapperOptsDefault, ...options};

        this.gpu = new GPU(opts.gpu);
        this.batchSize = opts.batchSize;

        this.layers = new Array(model.layers.length - 1);
        for (let i = 1; i < model.layers.length; i++) {
            this.layers[i - 1] = new GpuWrappedLayer(this.gpu, model.layers[i], i, this.batchSize);
        }

        this._lossCache = Matrix.zero_2d(opts.batchSize, model.outputSize);
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

    public train(input: Matrix2D, expected: Matrix2D, {epochs = 1} = {}) {
        this._assertNotDestroyed();

        for (let i = 0; i < epochs; i++) {
            this.model.beforeTrain();

            const shuffled = Iter.shuffled(Array.from(Iter.zip(input, expected)));
            for (const batch of Iter.partition(shuffled, this.batchSize)) {
                this.trainBatch(batch);
            }

            this.model.afterTrain();
        }
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
        if (input[0].length !== this.inputSize) throw new Error(`Wrong input dimension. Expected ${this.inputSize} got ${input[0].length}`);

        this.actualSize = input.length;
        let nexInput = input;
        for (const layer of this.layers) {
            nexInput = layer.forward(nexInput, this.actualSize);
        }

        return nexInput.slice(0, this.actualSize);
    }

    private backward(predicted: Matrix2D, expected: Matrix2D) {
        if (!this.actualSize) throw new Error("Not ready for backward pass");
        if (predicted.length > this.batchSize || expected.length > this.batchSize) throw new Error("Input/Output can't be greater than batchSize")
        if (predicted.length !== expected.length) throw new Error("Input and expected data sizes doesn't match!");
        if (predicted[0].length !== this.outputSize) throw new Error(`Wrong predicted dimension. Expected ${this.inputSize} got ${predicted[0].length}`);
        if (expected[0].length !== this.outputSize) throw new Error(`Wrong expected dimension. Expected ${this.outputSize} got ${expected[0].length}`);

        let nextError = predicted.map((p, i) => this.model.loss.calculateError(p, expected[i], this._lossCache[i]));
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];
            const res = layer.backward(nextError, this.actualSize);
            nextError = res.dError;

            if (this.model.isTrainable(layer.layer)) {
                this.model.optimizer.updateWeights(layer.layer, res.dW, res.dB, this.model.epoch, this.actualSize);
            }
        }

        this.actualSize = 0;
    }

    private _assertNotDestroyed() {
        if (this.isDestroyed) throw new Error("Wrapped destroyed");
    }
}
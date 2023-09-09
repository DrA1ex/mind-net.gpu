import {IModel} from "mind-net.js/engine/base";
import {ProgressOptions} from "mind-net.js/utils/progress"
import {Iter, ProgressUtils, DefaultTrainOpts} from "mind-net.js";
import {GPU, IGPUSettings} from "gpu.js";

import {GpuArray1D, GpuArray2D} from "./base";
import {GpuWrappedLayer} from "./layer";
import * as CommonUtils from "./utils/common";

export type GpuWrapperOptionsT = {
    batchSize: number,
    gpu?: IGPUSettings
}

export const GpuWrapperOptsDefault: GpuWrapperOptionsT = {
    batchSize: 128
}

export type GpuWrapperTrainOptionsT = {
    epochs: number,
    progress: boolean,
    progressOptions: Partial<ProgressOptions>
}

export const GpuWrapperTrainDefaultOpts: GpuWrapperTrainOptionsT = {
    epochs: DefaultTrainOpts.epochs,
    progress: DefaultTrainOpts.progress,
    progressOptions: DefaultTrainOpts.progressOptions
}

export class GpuModelWrapper {
    private readonly gpu: GPU;
    private actualSize: number = 0;
    private _destroyed = false;

    private readonly _lossCache: Float32Array[];
    private readonly _inputCache: Float32Array[];

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

        this._inputCache = CommonUtils.splitBatches(
            new Float32Array(this.batchSize * this.model.inputSize), this.model.inputSize)
        this._lossCache = CommonUtils.splitBatches(
            new Float32Array(this.batchSize * this.model.outputSize), this.model.outputSize);
    }

    public compute(input: GpuArray2D) {
        this._assertNotDestroyed();

        const result: GpuArray1D[] = [];
        for (const batch of Iter.partition(input, this.batchSize)) {
            const out = this.forward(batch, false);
            result.push(...out);
        }

        return result;
    }

    public train(input: GpuArray2D, expected: GpuArray2D, options: Partial<GpuWrapperTrainOptionsT> = {}) {
        this._assertNotDestroyed();

        const opts = {...GpuWrapperTrainDefaultOpts, ...options};
        const batchCtrl = opts.progress
            ? ProgressUtils.progressBatchCallback(input.length, opts.epochs, opts.progressOptions)
            : undefined;

        batchCtrl?.progress();
        for (let i = 0; i < opts.epochs; i++) {
            this.model.beforeTrain();

            const shuffled = Iter.shuffled(Array.from(Iter.zip(input, expected)));
            for (const batch of Iter.partition(shuffled, this.batchSize)) {
                this.trainBatch(batch);

                batchCtrl?.add(batch.length);
                batchCtrl?.progress();
            }

            this.model.afterTrain();
        }
    }

    public trainBatch(batch: [GpuArray1D, GpuArray1D][]) {
        this._assertNotDestroyed();

        const input = batch.map(b => b[0]);
        const expected = batch.map(b => b[1]);

        const predicted = this.forward(input, true);
        this.backward(predicted, expected);
    }

    destroy() {
        if (this.isDestroyed) return;

        for (const layer of this.layers) {
            layer.destroy();
        }

        this._destroyed = true;
    }

    private forward(input: GpuArray2D, isTraining: boolean): GpuArray2D {
        if (input.length > this.batchSize) throw new Error("Input can't be greater than batchSize")
        if (input[0].length !== this.inputSize) throw new Error(`Wrong input dimension. Expected ${this.inputSize} got ${input[0].length}`);

        CommonUtils.setArray(input, this._inputCache);

        this.actualSize = input.length;
        let nexInput: GpuArray2D = this._inputCache;
        for (const layer of this.layers) {
            nexInput = layer.forward(nexInput, this.actualSize, isTraining);
        }

        return nexInput.slice(0, this.actualSize);
    }

    private backward(predicted: GpuArray2D, expected: GpuArray2D) {
        if (!this.actualSize) throw new Error("Not ready for backward pass");
        if (predicted.length > this.batchSize || expected.length > this.batchSize) throw new Error("Input/Output can't be greater than batchSize")
        if (predicted.length !== expected.length) throw new Error("Input and expected data sizes doesn't match!");
        if (predicted[0].length !== this.outputSize) throw new Error(`Wrong predicted dimension. Expected ${this.inputSize} got ${predicted[0].length}`);
        if (expected[0].length !== this.outputSize) throw new Error(`Wrong expected dimension. Expected ${this.outputSize} got ${expected[0].length}`);

        for (let i = 0; i < predicted.length; i++) {
            this.model.loss.calculateError(predicted[i], expected[i], this._lossCache[i]);
        }

        let nextError: GpuArray2D = this._lossCache;
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
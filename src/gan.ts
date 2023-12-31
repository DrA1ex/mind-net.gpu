import {ProgressFn} from "mind-net.js/utils/fetch";
import {GenerativeAdversarialModel, Iter, Matrix, ProgressUtils} from "mind-net.js";

import {GpuArray1D, GpuArray2D} from "./base";
import {
    GpuModelWrapper,
    GpuWrapperOptionsT,
    GpuWrapperOptsDefault,
    GpuWrapperTrainDefaultOpts,
    GpuWrapperTrainOptionsT
} from "./model";

export class GpuGanWrapper {
    readonly chain: GpuModelWrapper;
    readonly generator: GpuModelWrapper;
    readonly discriminator: GpuModelWrapper;

    get epoch() {return this.chain.model.epoch;}
    get batchSize() {return this.chain.batchSize;}

    constructor(public readonly gan: GenerativeAdversarialModel, options: Partial<GpuWrapperOptionsT> = {}) {
        const opts = {...GpuWrapperOptsDefault, ...options};

        this.generator = new GpuModelWrapper(gan.generator, opts);
        this.discriminator = new GpuModelWrapper(gan.discriminator, {...opts, batchSize: opts.batchSize * 2});
        this.chain = new GpuModelWrapper(gan.ganChain, opts);
    }

    public compute(input: GpuArray2D): GpuArray2D {
        return this.generator.compute(input);
    }

    public train(real: GpuArray1D[], options: Partial<GpuWrapperTrainOptionsT> = {}) {
        const opts = {...GpuWrapperTrainDefaultOpts, ...options};

        const batchCtrl = opts.progress
            ? ProgressUtils.progressBatchCallback(
                3, Math.ceil(real.length / this.batchSize) * opts.epochs, opts.progressOptions
            ) : undefined;

        batchCtrl?.progress();
        for (let i = 0; i < opts.epochs; i++) {
            this.beforeTrain();

            const shuffledTrainSet = Iter.shuffled(real);
            for (const batch of Iter.partition(shuffledTrainSet, this.batchSize)) {
                this.trainBatch(batch, batchCtrl?.progressFn);
                batchCtrl?.addBatch();
            }

            this.afterTrain();
        }
    }

    public trainBatch(batch: GpuArray1D[], progressFn?: ProgressFn) {
        const totalProgress = 3;
        let currentBatch = 0;

        const almostOnes = Matrix.fill_value([0.9], batch.length);
        const zeros = Matrix.fill_value([0], batch.length);
        const noise = Matrix.random_normal_2d(batch.length, this.generator.inputSize, -1, 1);

        const fake = this.generator.compute(noise);
        if (progressFn) progressFn(++currentBatch, totalProgress);

        this.discriminator.trainBatch(
            Array.from(Iter.zip_iter(
                Iter.join(batch, fake),
                Iter.join(almostOnes, zeros)
            ))
        );

        if (progressFn) progressFn(++currentBatch, totalProgress);

        const trainNoise = Matrix.random_normal_2d(batch.length, this.generator.inputSize, -1, 1);
        const ones = Matrix.fill_value([1], batch.length);
        this.chain.trainBatch(Array.from(Iter.zip(trainNoise, ones)));

        if (progressFn) progressFn(++currentBatch, totalProgress);
    }

    public beforeTrain() {
        this.gan.beforeTrain();
    }

    public afterTrain() {
        this.gan.afterTrain();
    }

    public destroy() {
        this.generator.destroy();
        this.discriminator.destroy();
        this.chain.destroy();
    }
}
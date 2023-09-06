import {Matrix1D, Matrix2D} from "mind-net.js/engine/matrix";
import {GenerativeAdversarialModel, Iter, Matrix} from "mind-net.js";

import {GpuModelWrapper, GpuWrapperOptionsT, GpuWrapperOptsDefault} from "./model";

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

    public compute(input: Matrix2D): Matrix2D {
        return this.generator.compute(input);
    }

    public train(real: Matrix1D[], {epochs = 1} = {}) {
        for (let i = 0; i < epochs; i++) {
            this.beforeTrain();

            const shuffledTrainSet = Iter.shuffled(real);
            for (const batch of Iter.partition(shuffledTrainSet, this.batchSize)) {
                this.trainBatch(batch);
            }

            this.afterTrain();
        }
    }

    public trainBatch(batch: Matrix1D[]) {
        const almostOnes = Matrix.fill_value([0.9], batch.length);
        const zeros = Matrix.fill_value([0], batch.length);
        const noise = Matrix.random_normal_2d(batch.length, this.generator.inputSize, -1, 1);

        const fake = this.generator.compute(noise);

        this.discriminator.trainBatch(
            Array.from(Iter.zip_iter(
                Iter.join(batch, fake),
                Iter.join(almostOnes, zeros)
            ))
        );

        const trainNoise = Matrix.random_normal_2d(batch.length, this.generator.inputSize, -1, 1);
        const ones = Matrix.fill_value([1], batch.length);
        this.chain.trainBatch(Array.from(Iter.zip(trainNoise, ones)));
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
import {
    Dense,
    SequentialModel,
    ModelSerialization,
    GenerativeAdversarialModel,
    GanSerialization,
    Matrix
} from "mind-net.js";
import {IGPUSettings} from "gpu.js";

import {GpuGanWrapper, GpuModelWrapper} from "../src";

import {SetupMockRandom} from "./mock/common";
import * as ArrayUtils from "./utils/array";
import {GpuWrapperTrainDefaultOpts} from "../src/model";

function _simpleModel(optimizer = "sgd", loss = "mse") {
    const model = new SequentialModel(optimizer as any, loss as any)
        .addLayer(new Dense(3))
        .addLayer(new Dense(2));
    model.compile();
    return model;
}

const gpuOptions: IGPUSettings = {mode: "cpu"};
const eps = 1e-5;

const rndMock = SetupMockRandom([0.1, 0.2, 0.3, 0.2, 0.1], true)

// Disable progress
let origOptParallelTrain = GpuWrapperTrainDefaultOpts.progress;
beforeAll(() => GpuWrapperTrainDefaultOpts.progress = false);
afterAll(() => GpuWrapperTrainDefaultOpts.progress = origOptParallelTrain);

describe("Should correctly compute model", () => {
    test.each([1, 4, 8, 32])
    ("BatchSize: %d", batchSize => {
        const model = _simpleModel();
        const wrapped = new GpuModelWrapper(model, {batchSize, gpu: gpuOptions});

        const input1 = Matrix.random_2d(1, model.inputSize);
        const input2 = Matrix.random_2d(5, model.inputSize);
        const input3 = Matrix.random_2d(10, model.inputSize);

        ArrayUtils.arrayCloseTo_2d(wrapped.compute(input1), input1.map(v => model.compute(v)), eps);
        ArrayUtils.arrayCloseTo_2d(wrapped.compute(input2), input2.map(v => model.compute(v)), eps);
        ArrayUtils.arrayCloseTo_2d(wrapped.compute(input3), input3.map(v => model.compute(v)), eps);
    });
});

describe("Should correctly train model", () => {
    describe.each(["sgd", "rmsprop", "adam"])
    ("%p", () => {
        test.each([1, 4, 8, 32])
        ("BatchSize: %d", batchSize => {
            const model = _simpleModel();
            const modelCopy = ModelSerialization.load(ModelSerialization.save(model));
            const wrapped = new GpuModelWrapper(modelCopy, {batchSize, gpu: gpuOptions});

            const input = Matrix.random_2d(8, model.inputSize);
            const output = Matrix.random_2d(8, model.outputSize);

            rndMock.reset();
            model.train(input, output, {batchSize, epochs: 10, progress: false});

            rndMock.reset();
            wrapped.train(input, output, {epochs: 10});

            const test = Matrix.random_1d(model.inputSize);

            ArrayUtils.arrayCloseTo(wrapped.compute([test])[0], model.compute(test), eps);
        });
    });

    describe("Should correctly apply regularization", () => {
        test.each([
            {},
            {l1BiasRegularization: 0.1, l1WeightRegularization: 0.2},
            {l2BiasRegularization: 0.2, l2WeightRegularization: 0.3},
            {l1BiasRegularization: 0.01, l2BiasRegularization: 0.02},
            {l1WeightRegularization: 0.0004, l2BiasRegularization: 0.0003},
            {dropout: 0.1, l1WeightRegularization: 0.04},
            {dropout: 0.5},
        ])
        ("%p", (options) => {
            const batchSize = 8;
            const model = new SequentialModel()
                .addLayer(new Dense(3))
                .addLayer(new Dense(5, {options}))
                .addLayer(new Dense(6, {options}));
            model.compile();

            const modelCopy = ModelSerialization.load(ModelSerialization.save(model));
            const wrapped = new GpuModelWrapper(modelCopy, {batchSize, gpu: gpuOptions});

            const input = Matrix.random_2d(batchSize, model.inputSize);
            const output = Matrix.random_2d(batchSize, model.outputSize);

            rndMock.reset();
            model.train(input, output, {batchSize, epochs: 10, progress: false});

            rndMock.reset();
            wrapped.train(input, output, {epochs: 10});

            const test = Matrix.random_1d(model.inputSize);
            ArrayUtils.arrayCloseTo(wrapped.compute([test])[0], model.compute(test), eps);
        })
    });
});

describe("GpuGanWrapper should correctly wraps GAN model", () => {
    test.each([1, 2, 8, 16])
    ("%d", batchSize => {
        const generator = new SequentialModel()
            .addLayer(new Dense(3))
            .addLayer(new Dense(4));

        const discriminator = new SequentialModel()
            .addLayer(new Dense(4))
            .addLayer(new Dense(3))
            .addLayer(new Dense(1));

        const gan = new GenerativeAdversarialModel(generator, discriminator);

        const ganCopy = GanSerialization.load(GanSerialization.save(gan));
        const gpuGan = new GpuGanWrapper(ganCopy, {batchSize, gpu: gpuOptions});

        const input = Matrix.random_2d(8, discriminator.inputSize);

        rndMock.reset();
        gan.train(input, {batchSize, epochs: 10});

        rndMock.reset();
        gpuGan.train(input, {epochs: 10});

        const test = Matrix.random_1d(generator.inputSize);
        const generated = generator.compute(test);
        ArrayUtils.arrayCloseTo(generated, gpuGan.compute([test])[0], eps);
        ArrayUtils.arrayCloseTo(discriminator.compute(generated), ganCopy.discriminator.compute(generated), eps);
    });
});
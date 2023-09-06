import {Dense, SequentialModel, ModelSerialization, Matrix} from "mind-net.js";
import {IGPUSettings} from "gpu.js";

import {GpuModelWrapper} from "../src";

import {SetupMockRandom} from "./mock/common";
import * as ArrayUtils from "./utils/array";

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
});
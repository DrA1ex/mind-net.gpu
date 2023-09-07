# mind-net.gpu
Unleash the ⚡️GPU power⚡️ for [mind-net.js](https://www.npmjs.com/package/mind-net.js).

## Get Started

1. Install packages

```shell
npm install mind-net.js
npm install @mind-net.js/gpu
```

_Optionally_, if you encounter any build issues, you can add overrides for the gl package by modifying your package.json configuration as follows: 
```javascript
{
    //...
    "overrides": {
        "gl": "^6.0.2"
    }
}
```

2. Use imported binding
```javascript
import {SequentialModel, Dense} from "mind-net.js";
import {GpuModelWrapper} from "@mind-net.js/gpu";

const network = new SequentialModel();
network.addLayer(new Dense(2));
network.addLayer(new Dense(64, {activation: "leakyRelu"}));
network.addLayer(new Dense(1, {activation: "linear"}));
network.compile();

// Define the input and expected output data
const input = [[1, 2], [3, 4], [5, 6]];
const expected = [[3], [7], [11]];

// Create GPU wrapper
const batchSize = 128; // Note: batchSize specified only when creating the wrapper
const gpuWrapper = new GpuModelWrapper(network, batchSize);

// Train model
gpuWrapper.train(input, expected);

// Compute predictions
const predictions = gpuWrapper.compute(input);

// Free resources
gpuWrapper.destroy();
```
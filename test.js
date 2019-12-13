const NeuralNetwork = require('./src/neural-network-gpu');
const NeuralNetworkGPU = require('./src/neural-network-gpu');
const { FeedForward } = require('./src/feed-forward');
const { input, feedForward, target } = require('./src/layer');
const { GPU } = require('gpu.js');
const { setup } = require('./src/utilities/kernel');
const { setLogInternalWeights, setLogInternalDeltas, setCheckWeights, setCheckDeltas, findDeviationOrigin } = require('./src/layer/debug/strict-watch');

const xorTrainingData = [
  { input: [0, 1], output: [1] },
  { input: [0, 0], output: [0] },
  { input: [1, 1], output: [0] },
  { input: [1, 0], output: [1] }
];

const fs = require('fs');
try {
  fs.unlinkSync(`cpu-output.log`);
} catch (e) {}
try {
  fs.unlinkSync(`gpu-output.log`);
} catch (e) {}
let netType = null;
function demoXOR(net) {
  const status = net.train(xorTrainingData, {
    iterations: 400,
    errorThresh: 0.01,
    log: true,
    logPeriod: 100,
    callbackPeriod: 1,
    callback: (v) => {
      switch (v.from) {
        case 'runInput':
        case 'calculateDeltas':
        case 'adjustWeights':
          fs.appendFileSync(`${netType}-output.log`, 'FROM ' + v.from + '\n' + JSON.stringify(net.toJSON(), null, 2));
      }
    }
  });
  console.log(status);
  console.log(net.run([0,1]));
  console.log(net.run([0,0]));
  console.log(net.run([1,1]));
  console.log(net.run([1,0]));
}

// demoXOR(new NeuralNetwork());
// demoXOR(new NeuralNetworkGPU());

const { random } = require('./src/layer/random');
const { add } = require('./src/layer/add');
const { multiply } = require('./src/layer/multiply');
const { sigmoid } = require('./src/layer/sigmoid');

let layerWeights = null;
function setupWeights() {
  layerWeights = [
    [
      [
        -0.05144326761364937,
        0.039105065166950226
      ],
      [
        -0.1367069035768509,
        0.14960336685180664
      ],
      [
        0.14404159784317017,
        -0.01664450950920581
      ]
    ],
    [
      [0.0488249845802784],
      [-0.13172388076782227],
      [0.16559797525405884]
    ],
    [
      [
        -0.010835444554686546,
        0.0649796798825264,
        0.13607461750507355
      ]
    ],
    [
      [-0.1149999126791954]
    ]
  ];
}

function feedForward2(settings, input) {
  const { height } = settings;
  const weights = random({ name: 'weights', height, width: input.height });
  weights.weights = layerWeights.shift();
  const biases = random({ name: 'biases', height });
  biases.weights = layerWeights.shift();
  return sigmoid(add(multiply(weights, input), biases));
}

setupWeights();

setup(new GPU({ mode: netType = 'cpu' }));
const cpuNet = new FeedForward({
  inputLayer: () => input({ height: 2 }),
  hiddenLayers: [
    inputLayer => feedForward2({ height: 3 }, inputLayer),
    inputLayer => feedForward2({ height: 1 }, inputLayer),
  ],
  outputLayer: inputLayer => target({ height: 1 }, inputLayer),
  praxisOpts: {
    decayRate: 0.99
  }
});
const cpuInitialize = cpuNet.initialize;
cpuNet.initialize = function() {
  setLogInternalWeights(false);
  setLogInternalDeltas(false);
  cpuInitialize.call(this);
  setLogInternalWeights(true);
  setLogInternalDeltas(true);
};
demoXOR(cpuNet);
setLogInternalWeights(false);
setLogInternalDeltas(false);
setupWeights();
setup(new GPU({ mode: netType = 'gpu' }));
const gpuNet = new FeedForward({
  inputLayer: () => input({ height: 2 }),
  hiddenLayers: [
    inputLayer => feedForward2({ height: 3 }, inputLayer),
    inputLayer => feedForward2({ height: 1 }, inputLayer),
  ],
  outputLayer: inputLayer => target({ height: 1 }, inputLayer),
  praxisOpts: {
    decayRate: 0.99
  }
});
// const gpuInitialize = gpuNet.initialize;
// gpuNet.initialize = function() {
//   gpuInitialize.call(this);
  // setCheckWeights(checkDeviation);
  // setCheckDeltas(checkDeviation);
// };
demoXOR(gpuNet);

// function checkDeviation(cpuResult, gpuResult) {
//   const threshold = 0.001;
//   if (cpuResult[0][0] instanceof Float32Array) {
//     const z = cpuResult.length;
//     const y = cpuResult[0].length;
//     const x = cpuResult[0][0].length;
//     for (let zIndex = 0; zIndex < z; zIndex++) {
//       for (let yIndex = 0; yIndex < y; yIndex++) {
//         for (let xIndex = 0; xIndex < x; xIndex++) {
//           if (Math.abs(gpuResult[zIndex][yIndex][xIndex] - cpuResult[zIndex][yIndex][xIndex]) >= threshold) {
//             findDeviationOrigin();
//             throw new Error('deviation!');
//           }
//         }
//       }
//     }
//   } else if (cpuResult[0] instanceof Float32Array) {
//     const y = cpuResult.length;
//     const x = cpuResult[0].length;
//     for (let yIndex = 0; yIndex < y; yIndex++) {
//       for (let xIndex = 0; xIndex < x; xIndex++) {
//         if (Math.abs(gpuResult[yIndex][xIndex] - cpuResult[yIndex][xIndex]) >= threshold) {
//           findDeviationOrigin();
//           throw new Error('deviation!');
//         }
//       }
//     }
//   } else {
//     const x = cpuResult.length;
//     for (let xIndex = 0; xIndex < x; xIndex++) {
//       if (Math.abs(gpuResult[xIndex] - cpuResult[xIndex]) >= threshold) {
//         findDeviationOrigin();
//         throw new Error('deviation!');
//       }
//     }
//   }
//   return true;
// }

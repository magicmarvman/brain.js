const { makeKernel } = require('../utilities/kernel');
const zeros2D = require('../utilities/zeros-2d');

const { Base } = require('./base');

function setXSum(xsum) {
  return xsum;
}

function setGSum(gsum) {
  return gsum;
}

function update(weights, deltas, gSums, xSums, decayRates) {
  const weight = weights[this.thread.y][this.thread.x];
  const delta = deltas[this.thread.y][this.thread.x];
  const learningGradient1 = decayRates[0] * (weight > 0 ? 1 : -1);
  const learningGradient2 = decayRates[1] * weight;
  const rawBatchGradient = (learningGradient2 + learningGradient1 + delta) / this.constants.batchSize;
  const gSum = this.constants.ro * gSums[this.thread.y][this.thread.x] + (1 - this.constants.ro) * rawBatchGradient * rawBatchGradient;
  const xSum = xSums[this.thread.y][this.thread.x];
  const dx = - Math.sqrt((xSum + this.constants.eps) / (gSum + this.constants.eps)) * rawBatchGradient;
  setGSum(gSum);
  setXSum(this.constants.ro * xSum + (1 - this.constants.ro) * dx * dx);
  return dx;
}

function updateDecayRates(decayRates, decayMultipliers) {
  return decayRates[this.thread.x] * decayMultipliers[this.thread.x];
}

class Adadelta extends Base {
  static get defaults() {
    return {
      eps: 1e-6,
      ro: 0.95,
      batchSize: 1,
      decayRates: [0, 0],
      decayMultipliers: [0, 1],
    };
  }

  constructor(layerTemplate, settings) {
    super(layerTemplate, settings);
    this.updateDecayRatesKernel = null;
    this.gSums = zeros2D(this.width, this.height);
    this.xSums = zeros2D(this.width, this.height);
    this.setupKernels();
  }

  setupKernels() {
    this.kernel = makeKernel(update, {
      output: [this.width, this.height],
      constants: {
        batchSize: this.batchSize,
        ro: this.ro,
        eps: this.eps,
      },
      map: {
        gSums: setGSum,
        xSums: setXSum,
      }
    });
    this.updateDecayRatesKernel = makeKernel(updateDecayRates, {
      output: [2]
    });
  }

  run(layer) {
    const decayRates = this.updateDecayRatesKernel(this.decayRates, this.decayMultipliers);
    this.decayRates = decayRates;
    const { xSums, gSums, result } = this.kernel(layer.weights, layer.deltas, this.gSums, this.xSums, decayRates);
    this.xSums = xSums;
    this.gSums = gSums;
    return result;
  }
}

module.exports = {
  Adadelta
};

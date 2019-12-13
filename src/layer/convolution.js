const { makeKernel } = require('../utilities/kernel');
const { setStride, setPadding } = require('../utilities/layer-setup');
const { Filter } = require('./types');
const randos = require('../utilities/randos');
const randos3D = require('../utilities/randos-3d');
const zeros3D = require('../utilities/zeros-3d');
const randos2D = require('../utilities/randos-2d');
const zeros2D = require('../utilities/zeros-2d');
const values = require('../utilities/values');

function predict2D(inputs, filters, biases) {
  const startFilterX = this.constants.paddingX - (this.thread.x * this.constants.strideX);
  const startInputX = (this.thread.x * this.constants.strideX) - this.constants.paddingX;
  const endFilterX = Math.min(this.constants.filterWidth, startFilterX + this.constants.inputWidth);

  const startFilterY = this.constants.paddingY - (this.thread.y * this.constants.strideY);
  const startInputY = (this.thread.y * this.constants.strideY) - this.constants.paddingY;
  const endFilterY = Math.min(this.constants.filterHeight, startFilterY + this.constants.inputHeight);

  let sum = 0;
  for (let filterY = Math.max(0, startFilterY), inputY = Math.max(0, startInputY); filterY < endFilterY; filterY++, inputY++) {
    for (let filterX = Math.max(0, startFilterX), inputX = Math.max(0, startInputX); filterX < endFilterX; filterX++, inputX++) {
      sum += filters[filterY][filterX] * inputs[inputY][inputX];
    }
  }
  return sum + biases[this.thread.z];
}

function predict3D(inputs, filters, biases) {
  const startFilterX = this.constants.paddingX - (this.thread.x * this.constants.strideX);
  const startInputX = (this.thread.x * this.constants.strideX) - this.constants.paddingX;
  const endFilterX = Math.min(this.constants.filterWidth, startFilterX + this.constants.inputWidth);

  const startFilterY = this.constants.paddingY - (this.thread.y * this.constants.strideY);
  const startInputY = (this.thread.y * this.constants.strideY) - this.constants.paddingY;
  const endFilterY = Math.min(this.constants.filterHeight, startFilterY + this.constants.inputHeight);

  let sum = 0;
  for (let z = 0; z < this.constants.inputDepth; z++) {
    for (let filterY = Math.max(0, startFilterY), inputY = Math.max(0, startInputY); filterY < endFilterY; filterY++, inputY++) {
      for (let filterX = Math.max(0, startFilterX), inputX = Math.max(0, startInputX); filterX < endFilterX; filterX++, inputX++) {
        sum += filters[z][filterY][filterX] * inputs[z][inputY][inputX];
      }
    }
  }
  return sum + biases[this.thread.z];
}

function compareFilterDeltas(filterDeltas, inputs, deltas, deltaZ) {
  const startDeltaX = Math.max(0, Math.ceil((this.constants.paddingX - this.thread.x) / this.constants.strideX));
  const startInputX = startDeltaX * this.constants.strideX + this.thread.x - this.constants.paddingX;
  const endDeltaX = Math.min(this.constants.deltaWidth, Math.floor(((this.constants.inputWidth - 1) - this.thread.x + this.constants.paddingX) / this.constants.strideX) + 1);

  const startDeltaY = Math.max(0, Math.ceil((this.constants.paddingY - this.thread.y) / this.constants.strideY));
  const startInputY = startDeltaY * this.constants.strideY + this.thread.y - this.constants.paddingY;
  const endDeltaY = Math.min(this.constants.deltaHeight, Math.floor(((this.constants.inputHeight - 1) - this.thread.y + this.constants.paddingY) / this.constants.strideY) + 1);

  let sum = filterDeltas[this.thread.z][this.thread.y][this.thread.x];
  for (let deltaY = startDeltaY, inputY = startInputY; deltaY < endDeltaY; deltaY++, inputY += this.constants.strideY) {
    for (let deltaX = startDeltaX, inputX = startInputX; deltaX < endDeltaX; deltaX++, inputX += this.constants.strideX) {
      sum += inputs[this.thread.z][inputY][inputX] * deltas[deltaZ][deltaY][deltaX];
    }
  }
  return sum;
}

function compareInputDeltas(inputDeltas, filters, deltas, deltaZ) {
  const x = this.thread.x + this.constants.paddingX;
  const startDeltaX = x < this.constants.filterWidth ? 0 : Math.floor((x - this.constants.filterWidth + this.constants.strideX) / this.constants.strideX);
  const startFilterX = x - startDeltaX * this.constants.strideX;
  const endDeltaX = Math.min(startDeltaX + Math.floor(startFilterX / this.constants.strideX) + 1, this.constants.deltaWidth);

  const y = this.thread.y + this.constants.paddingY;
  const startDeltaY = y < this.constants.filterHeight ? 0 : Math.floor((y - this.constants.filterHeight + this.constants.strideY) / this.constants.strideY);
  const startFilterY = y - startDeltaY * this.constants.strideY;
  const endDeltaY = Math.min(startDeltaY + Math.floor(startFilterY / this.constants.strideY) + 1, this.constants.deltaHeight);

  let sum = inputDeltas[this.thread.z][this.thread.y][this.thread.x];
  let deltaY = startDeltaY;
  for (let filterY = startFilterY; deltaY < endDeltaY; filterY -= this.constants.strideY, deltaY++) {
    let deltaX = startDeltaX;
    for (let filterX = startFilterX; deltaX < endDeltaX; filterX -= this.constants.strideX, deltaX++) {
      sum += filters[this.thread.z][filterY][filterX] * deltas[deltaZ][deltaY][deltaX];
    }
  }
  return sum;
}

function compareBiases(biasDeltas, deltas) {
  let sum = 0;
  for (let y = 0; y < this.constants.deltaHeight; y++) {
    for (let x = 0; x < this.constants.deltaWidth; x++) {
      sum += deltas[this.thread.z][y][x];
    }
  }
  return biasDeltas[this.thread.z][this.thread.y][this.thread.x] + sum;
}

class Convolution extends Filter {
  static get defaults() {
    return {
      stride: 0,
      padding: 0,
      bias: 0.1,
      filterCount: 1,
      filterWidth: 1,
      filterHeight: 1,
    };
  }

  constructor(settings, inputLayer) {
    super(settings);

    this.stride = null;
    this.strideX = null;
    this.strideY = null;
    setStride(this, settings);

    this.padding = null;
    this.paddingX = null;
    this.paddingY = null;
    setPadding(this, settings);

    this.filterCount = settings.filterCount;
    this.filterWidth = settings.filterWidth;
    this.filterHeight = settings.filterHeight;

    this.width = Math.floor(
      (inputLayer.width + this.paddingX * 2 - this.filterWidth) / this.strideX +
        1
    );
    this.height = Math.floor(
      (inputLayer.height + this.paddingY * 2 - this.filterHeight) /
        this.strideY +
        1
    );
    this.depth = this.filterCount;
    this.weights = randos3D(this.width, this.height, this.depth);
    this.deltas = zeros3D(this.width, this.height, this.depth);

    this.biases = values(this.depth, this.bias);
    this.biasDeltas = randos(this.depth);

    this.filters = null;
    this.filterDeltas = null;
    this.filterPraxis = null;
    this.learnFilters = null;
    this.learnInputs = null;
    this.inputLayer = inputLayer;
    this.validate();
    this.setupPraxis(settings);
    this.setupFilters();
    this.setupFilterPraxis(settings);
  }

  setupFilters() {
    this.filters = [];
    this.filterDeltas = [];
    if (this.inputLayer.depth) {
      for (let i = 0; i < this.filterCount; i++) {
        this.filters.push(randos3D(this.filterWidth, this.filterHeight, this.inputLayer.depth));
        this.filterDeltas.push(zeros3D(this.filterWidth, this.filterHeight, this.inputLayer.depth));
      }
    } else {
      for (let i = 0; i < this.filterCount; i++) {
        this.filters.push(randos2D(this.filterWidth, this.filterHeight));
        this.filterDeltas.push(zeros2D(this.filterWidth, this.filterHeight));
      }
    }
  }

  setupFilterPraxis(settings) {
    if (!settings) return;
    if (settings.hasOwnProperty('praxis')) {
      if (typeof settings.praxis === 'function') {
        this.praxis = settings.praxis({ weights: this.filters[0], deltas: this.filterDeltas[0] }, settings.praxisOpts);
      } else {
        this.praxis = settings.praxis;
      }
    }
  }

  setupKernels() {
    this.predictKernel = makeKernel(this.inputLayer.depth ? predict3D : predict2D, {
      constants: {
        inputWidth: this.inputLayer.width,
        inputHeight: this.inputLayer.height,
        inputDepth: this.inputLayer.depth,
        strideX: this.strideX,
        strideY: this.strideY,
        paddingX: this.paddingX,
        paddingY: this.paddingY,
        filterWidth: this.filterWidth,
        filterHeight: this.filterHeight,
      },
      output: [this.width, this.height, this.depth],
    });

    this.compareFilterDeltasKernel = makeKernel(compareFilterDeltas, {
      constants: {
        deltasWidth: this.width,
        deltasHeight: this.height,
        deltasDepth: this.depth,
        inputWidth: this.inputLayer.width,
        inputHeight: this.inputLayer.height,
        inputDepth: this.inputLayer.depth,
        strideX: this.strideX,
        strideY: this.strideY,
        paddingX: this.paddingX,
        paddingY: this.paddingY,
        filterWidth: this.filterWidth,
        filterHeight: this.filterHeight,
      },
      output: this.inputLayer.depth
        ? [this.filterWidth, this.filterHeight, this.inputLayer.depth]
        : [this.filterWidth, this.filterHeight],
    });

    this.compareInputDeltasKernel = makeKernel(compareInputDeltas, {
      constants: {
        filterCount: this.filterCount,
      },
      output: [
        this.inputLayer.width,
        this.inputLayer.height,
        this.inputLayer.depth,
      ],
    });

    this.compareBiasesKernel = makeKernel(compareBiases, {
      output: [1, 1, this.depth],
      constants: {
        deltaWidth: this.width,
        deltaHeight: this.height,
      },
    });
  }

  predict() {
    this.weights = this.predictKernel(
      this.inputLayer.weights,
      this.filters,
      this.biases
    );
  }

  compare() {
    for (let i = 0; i < this.filterDeltas.length; i++) {
      const filterDeltasClone = this.filterDeltas[i].clone();
      this.filterDeltas[i] = this.compareFilterDeltasKernel(
        filterDeltasClone,
        this.inputLayer.weights,
        this.deltas,
        i
      );
      filterDeltasClone.delete();
    }
    this.biasDeltas = this.compareBiasesKernel(this.biasDeltas, this.deltas);
    this.deltas = this.compareInputDeltasKernel(this.filters, this.inputLayer.deltas);
    this.inputLayer.deltas = this.deltas;
  }

  learn(learningRate) {
    // TODO: handle filters and biases
    this.weights = this.praxis.run(this, learningRate);
    const virtualLayer = {};
    for (let i = 0; i < this.filterCount.length; i++) {
      virtualLayer.weights = this.filters[i].copy();
      virtualLayer.deltas = this.filterDeltas[i];
      this.filters[i] = this.filterPraxis.run(virtualLayer, learningRate);
      virtualLayer.weights.delete();
    }
    this.deltas = zeros3D(this.width, this.height, this.depth);
  }
}

function convolution(settings, inputLayer) {
  return new Convolution(settings, inputLayer);
}

module.exports = {
  Convolution,
  convolution,
  predict2D,
  predict3D,
  compareFilterDeltas,
  compareInputDeltas,
  compareBiases
};

const { GPU } = require('gpu.js');

class DeviationTester {
  constructor() {
    this.threshold = 0.00001;
    this.cpu = new GPU({ mode: 'cpu' });
    this.gpu = new GPU({ mode: 'gpu' });
  }

  createKernelMap(map, fn, settings) {
    const { threshold, gpu, cpu } = this;
    const cpuKernel = cpu.createKernelMap(map, fn, settings);
    const gpuKernel = gpu.createKernelMap(map, fn, settings);

    function result() {
      const [x, y, z] = gpuKernel.output;
      const cpuResult = cpuKernel.apply(this, Array.from(arguments).map(getCPUValue));
      const gpuTexture = gpuKernel.apply(this, arguments);

      for (const p in map) {
        const gpuResult = gpuTexture[p].toArray();
        gpuTexture[p].cpuResult = cpuResult[p];
        checkDeviation(cpuResult[p], gpuResult, threshold, x, y, z);
      }

      return gpuTexture;
    }
    addMethods(result, cpuKernel, gpuKernel);
    return result;
  }

  createKernel(fn, settings) {
    const { threshold, gpu, cpu } = this;
    let cpuKernel = cpu.createKernel(fn, settings);
    const gpuKernel = gpu.createKernel(fn, settings);
    function result() {
      const [x, y, z] = gpuKernel.output;
      const cpuResult = cpuKernel.apply(this, Array.from(arguments).map(getCPUValue));
      const gpuTexture = gpuKernel.apply(this, arguments);

      if (gpuTexture.toArray) {
        const gpuResult = gpuTexture.toArray();
        checkDeviation(cpuResult, gpuResult, threshold, x, y, z);
      }
      gpuTexture.cpuResult = cpuResult;
      return gpuTexture;
    }
    addMethods(result, cpuKernel, gpuKernel);
    return result;
  }
}

function checkDeviation(cpuResult, gpuResult, threshold, x, y, z) {
  if (z) {
    for (let zIndex = 0; zIndex < z; zIndex++) {
      for (let yIndex = 0; yIndex < y; yIndex++) {
        for (let xIndex = 0; xIndex < x; xIndex++) {
          if (Math.abs(gpuResult[zIndex][yIndex][xIndex] - cpuResult[zIndex][yIndex][xIndex]) >= threshold) {
            throw new Error('deviation!');
          }
        }
      }
    }
  } else if (y) {
    for (let yIndex = 0; yIndex < y; yIndex++) {
      for (let xIndex = 0; xIndex < x; xIndex++) {
        if (Math.abs(gpuResult[yIndex][xIndex] - cpuResult[yIndex][xIndex]) >= threshold) {
          throw new Error('deviation!');
        }
      }
    }
  } else {
    for (let xIndex = 0; xIndex < x; xIndex++) {
      if (Math.abs(gpuResult[xIndex] - cpuResult[xIndex]) >= threshold) {
        throw new Error('deviation!');
      }
    }
  }
  return true;
}

function addMethods(fn, cpuKernel, gpuKernel) {
  fn.setPipeline = function(flag) {
    gpuKernel.setPipeline(flag);
    cpuKernel.setPipeline(flag);
    return fn;
  };
}

function getCPUValue(value) {
  if (value.cpuResult) {
    return value.cpuResult;
  } else if (value.toArray) {
    return value.toArray();
  } else {
    return value;
  }
}

module.exports = { DeviationTester };

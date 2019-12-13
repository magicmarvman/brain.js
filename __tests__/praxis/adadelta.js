const { GPU } = require('gpu.js');

const { Adadelta } = require('../../src/praxis/adadelta');
const { setup, teardown } = require('../../src/utilities/kernel');
const { injectIstanbulCoverage } = require('../test-utils');

describe('Adadelta', () => {
  beforeEach(() => {
    setup(new GPU({
      mode: 'cpu',
      onIstanbulCoverageVariable: injectIstanbulCoverage
    }));
  });
  afterEach(() => {
    teardown();
  });
  describe('.run', () => {
    test('correctly runs values', () => {
      const layer = { weights: [[1]], deltas: [[1]], width: 1, height: 1 };
      const praxis = new Adadelta(layer, {
        eps: 1e-6,
        ro: 0.95,
        batchSize: 1,
        decayRates: [0, 0],
        decayMultipliers: [0, 1],
      });
      const result = praxis.run(layer);
      expect(result[0][0].toFixed(5)).toEqual('-0.00447');
      expect(praxis.xSums[0][0].toFixed(5)).toBe('0.00000');
      expect(praxis.gSums[0][0].toFixed(5)).toBe('0.05000');
    });
  });
});

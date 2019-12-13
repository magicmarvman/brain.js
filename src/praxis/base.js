class Base {
  static get defaults() {
    return {};
  }

  constructor(layerTemplate, settings = {}) {
    this.layer = layerTemplate;
    this.width = layerTemplate.width || null;
    this.height = layerTemplate.height || null;
    this.depth = layerTemplate.depth || null;
    Object.assign(this, this.constructor.defaults, settings);
  }

  setupKernels() {}

  /**
   *
   * @param {Base} layer
   * @param {Number} learningRate
   * @abstract
   */
  run(layer, learningRate) {}
}

module.exports = {
  Base
};

let logInternalWeights = false;
let logInternalDeltas = false;

let checkWeights = null;
let checkDeltas = null;

let checkedWeights = null;
let checkedDeltas = null;

const loggedWeights = [];
const loggedDeltas = [];

function setLogInternalWeights(value) {
  logInternalWeights = value;
}
function setLogInternalDeltas(value) {
  logInternalDeltas = value;
}

function setCheckWeights(value) {
  checkedWeights = [];
  checkWeights = value;
}

function setCheckDeltas(value) {
  checkedDeltas = [];
  checkDeltas = value;
}

function watchWeights(Class) {
  Object.defineProperty(Class.prototype, 'weights', {
    get() {
      return this._weights;
    },
    set(value) {
      if (value) {
        if (value.texture) {
          if (value.output[0] !== this.width) {
            throw new Error(`${this.constructor.name}.weights being set with improper value width`);
          }
          if (value.output[1] !== this.height) {
            throw new Error(`${this.constructor.name}.weights being set with improper value height`);
          }
        } else {
          if (value[0].length !== this.width) {
            throw new Error(`${this.constructor.name}.weights being set with improper value width`);
          }
          if (value.length !== this.height) {
            throw new Error(`${this.constructor.name}.weights being set with improper value height`);
          }
        }
      }
      if (value) {
        if (logInternalWeights) {
          loggedWeights.push(value.toArray ? value.toArray() : value);
        } else if (checkWeights) {
          checkWeights(loggedWeights[checkedWeights.length], value.toArray ? value.toArray() : value);
          checkedWeights.push(value);
        }
      }
      this._weights = value;
    }
  });
}

function watchDeltas(Class) {
  Object.defineProperty(Class.prototype, 'deltas', {
    get() {
      return this._deltas;
    },
    set(value) {
      if (value) {
        if (value.texture) {
          if (value.output[0] !== this.width) {
            throw new Error(`${this.constructor.name}.weights being set with improper value width`);
          }
          if (value.output[1] !== this.height) {
            throw new Error(`${this.constructor.name}.weights being set with improper value height`);
          }
        } else {
          if (value[0].length !== this.width) {
            throw new Error(`${this.constructor.name}.deltas being set with improper value width`);
          }
          if (value.length !== this.height) {
            throw new Error(`${this.constructor.name}.deltas being set with improper value height`);
          }
        }
      }
      if (value) {
        if (logInternalDeltas) {
          loggedDeltas.push(value.toArray ? value.toArray() : value);
        } else if (checkDeltas) {
          checkDeltas(loggedDeltas[checkedDeltas.length], value.toArray ? value.toArray() : value);
          checkedDeltas.push(value);
        }
      }
      this._deltas = value;
    }
  });
}

function findDeviationOrigin() {
  function checkValues(loggedValues, checkedValues) {
    const threshold = 0.001;
    for (let i = 0; i < checkedValues.length; i++) {
      const loggedValue = loggedValues[i];
      const checkedValue = checkedValues[i].toArray();
      if (loggedValue[0][0] instanceof Float32Array) {
        const z = loggedValue.length;
        const y = loggedValue[0].length;
        const x = loggedValue[0][0].length;
        for (let zIndex = 0; zIndex < z; zIndex++) {
          for (let yIndex = 0; yIndex < y; yIndex++) {
            for (let xIndex = 0; xIndex < x; xIndex++) {
              if (Math.abs(checkedValue[zIndex][yIndex][xIndex] - loggedValue[zIndex][yIndex][xIndex]) >= threshold) {
                throw new Error('deviation origin!');
              }
            }
          }
        }
      } else if (loggedValue[0] instanceof Float32Array) {
        const y = loggedValue.length;
        const x = loggedValue[0].length;
        for (let yIndex = 0; yIndex < y; yIndex++) {
          for (let xIndex = 0; xIndex < x; xIndex++) {
            if (Math.abs(checkedValue[yIndex][xIndex] - loggedValue[yIndex][xIndex]) >= threshold) {
              throw new Error('deviation origin!');
            }
          }
        }
      } else {
        const x = loggedValue.length;
        for (let xIndex = 0; xIndex < x; xIndex++) {
          if (Math.abs(checkedValue[xIndex] - loggedValue[xIndex]) >= threshold) {
            throw new Error('deviation origin!');
          }
        }
      }
    }
  }
  checkValues(loggedWeights, checkedWeights);
  checkValues(loggedDeltas, checkedDeltas);
}

module.exports = {
  watchWeights,
  watchDeltas,
  setLogInternalWeights,
  setLogInternalDeltas,
  setCheckWeights,
  setCheckDeltas,
  findDeviationOrigin,
};

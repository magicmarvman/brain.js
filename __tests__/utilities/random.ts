import  {randomFloat, randomInteger, randomN} from "../../src/utilities/random";

describe('random', () => {
  test('randomF', () => {
    const val: number = randomFloat(0, 10);

    expect(val).toBeGreaterThan(0);
    expect(val).toBeLessThan(11);
  });

  test('randomI', () => {
    const val: number = randomInteger(0, 10);

    expect(val).toBeGreaterThanOrEqual(0);
    expect(val).toBeLessThan(11);
  });

  test('randomN', () => {
    const val: number = randomN(10, 5);

    expect(typeof val).toBe('number');
  });
});

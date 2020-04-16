import numpy as np
import unittest

def discount_cumsum(x, dones, gamma):
    """
    computing discounted cumulative sums of vectors that resets with dones
    input:
        vector x,  vector dones,
        [x0,       [0,
         x1,        0,
         x2         1,
         x3         0, 
         x4]        0]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2,
         x3 + discount * x4,
         x4]
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1] * (1-dones[t])
    return discount_cumsum



class DiscountCumsumMethods(unittest.TestCase):

    def test_upper(self):
        x = np.array([1.,1.,1.,1.,1.])
        dones = np.array([0.,0.,1.,0.,0.])
        discount_cumsum(x, dones, 0.99)
        np.testing.assert_array_almost_equal(discount_cumsum(x, dones, 0.99), np.array([
            1 + 0.99 * 1 + 0.99**2 * 1,
            1 + 0.99 * 1,
            1,
            1 + 0.99 * 1,
            1
        ]))

if __name__ == '__main__':
    unittest.main()
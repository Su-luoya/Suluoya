class Loan(object):

    def __init__(self, L=100, n=120, R=0.006):
        self.L = L
        self.n = n
        self.R = R

    def remain(self, A):
        begin = self.L
        interest = begin*self.R
        repay = A - interest
        end = begin - repay
        for i in range(2, self.n+1):
            begin = end
            interest = begin*self.R
            repay = A - interest
            end = begin - repay
        return end

    def iter(self, step=0.01, error_control=0.1):
        error = 999999
        A = self.L/self.n
        t = 1
        while error > error_control:
            print(
                f'-----------------------------\n第{t}次迭代')
            t += 1
            remain = self.remain(A)
            print('remain', remain)
            if remain > 0:
                A += step
            elif remain < 0:
                A -= step
            error = abs(remain)/self.L
            print('error', error)
        return A


if __name__ == '__main__':
    l = Loan(L=100, n=120, R=0.006)
    result = l.iter(step=0.00001, error_control=0.00001)
    print(result)

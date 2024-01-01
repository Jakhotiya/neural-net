from neuron import Network

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    xs = [
        [1],
        [1.5],
        [1.2],
        [5.6],
        [9.0],
        [8.5],
        [0.1]
    ]
    yreal = [0, 0, 0,1 , 1,1,0]
    net = Network(1, [])


    def train():

        for i in range(5):

            # forward pass
            ypred = [net(x)[0] for x in xs]
            # calculate loss
            loss = sum((prediction - real) ** 2 for real, prediction in zip(yreal, ypred))

            # reinitialize grandients because we don't want these to be accumulated
            for p in net.parameters():
                p._grad = 0

            loss.backward()

            for p in net.parameters():
                p.data += -0.01 * p._grad
            print(loss)


    train()
    prediction = net([2])[0]

    print(prediction.data)



import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from population import Population


class psoCNN:
    def __init__(self, dataset, n_iter, pop_size, batch_size, epochs, min_layer, max_layer, conv_prob, pool_prob,
                 fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, dropout_rate, activation = 'relu'):

        self.pop_size = pop_size
        self.n_iter = n_iter
        self.epochs = epochs

        self.batch_size = batch_size
        self.gBest_acc = np.zeros(n_iter)
        self.gBest_test_acc = np.zeros(n_iter)

        self.activation = activation

        if dataset == "mnist":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "fashion-mnist":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)
            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_test.astype('float32')
            self.x_train /= 255
            self.x_test /= 255

        if dataset == "mnist-background-images":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("./datasets/mnist-background-images/mnist_background_images_train.amat")
            test = np.loadtxt("./datasets/mnist-background-images/mnist_background_images_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "mnist-rotated-digits":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("./datasets/mnist-rotated-digits/mnist_all_rotation_normalized_float_train_valid.amat")
            test = np.loadtxt("./datasets/mnist-rotated-digits/mnist_all_rotation_normalized_float_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "mnist-random-background":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("./datasets/mnist-random-background/mnist_background_random_train.amat")
            test = np.loadtxt("./datasets/mnist-random-background/mnist_background_random_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "mnist-rotated-with-background":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt(
                "./datasets/mnist-rotated-with-background/mnist_all_background_images_rotation_normalized_train_valid.amat")
            test = np.loadtxt(
                "./datasets/mnist-rotated-with-background/mnist_all_background_images_rotation_normalized_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "rectangles":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("./datasets/rectangles/rectangles_train.amat")
            test = np.loadtxt("./datasets/rectangles/rectangles_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "rectangles-images":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("./datasets/rectangles-images/rectangles_im_train.amat")
            test = np.loadtxt("./datasets/rectangles-images/rectangles_im_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "convex":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("./datasets/convex/convex_train.amat")
            test = np.loadtxt("./datasets/convex/convex_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "early-dataset":
            input_width = 256
            input_height = 64
            input_channels = 3  # RGB images
            output_dim = 4  # Assuming 4 classes based on your filename format

            train = np.loadtxt("./datasets/early-dataset/early_train.amat")
            test = np.loadtxt("./datasets/early-dataset/early_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 256x64x3
            self.x_train = np.reshape(self.x_train, (-1, input_height, input_width, input_channels))
            self.x_test = np.reshape(self.x_test, (-1, input_height, input_width, input_channels))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        if dataset == "late-color-dataset":
            input_width = 128
            input_height = 64
            input_channels = 3  # RGB images
            output_dim = 4  # Assuming 4 classes based on your filename format

            train = np.loadtxt("./datasets/late-color-dataset/late_color_train.amat")
            test = np.loadtxt("./datasets/late-color-dataset/late_color_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 256x64x3
            self.x_train = np.reshape(self.x_train, (-1, input_height, input_width, input_channels))
            self.x_test = np.reshape(self.x_test, (-1, input_height, input_width, input_channels))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

            print(f"DEBUG: In psoCNN.__init__ for dataset '{dataset}'")
            print(f"DEBUG: self.x_train shape: {self.x_train.shape}")
            print(f"DEBUG: self.y_train shape: {self.y_train.shape}")

        if dataset == "late-depth-dataset":
            input_width = 128
            input_height = 64
            input_channels = 3  # RGB images
            output_dim = 4  # Assuming 4 classes based on your filename format

            train = np.loadtxt("./datasets/late-depth-dataset/late_depth_train.amat")
            test = np.loadtxt("./datasets/late-depth-dataset/late_depth_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 256x64x3
            self.x_train = np.reshape(self.x_train, (-1, input_height, input_width, input_channels))
            self.x_test = np.reshape(self.x_test, (-1, input_height, input_width, input_channels))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]
            self.y_train = to_categorical(self.y_train, output_dim)
            self.y_test = to_categorical(self.y_test, output_dim)

        print("Initializing population...")
        self.population = Population(pop_size, min_layer, max_layer, input_width, input_height, input_channels,
                                     conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons,
                                     output_dim)

        print("Verifying accuracy of the current gBest...")
        print(self.population.particle[0])
        # Use safe cloning method instead of deepcopy
        self.gBest = self.population.particle[0].clone_particle()
        self.gBest.model_compile(dropout_rate)
        hist = self.gBest.model_fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
        test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)
        self.gBest.model_delete()

        self.gBest_acc[0] = hist.history['accuracy'][-1]
        self.gBest_test_acc[0] = test_metrics[1]

        self.population.particle[0].acc = hist.history['accuracy'][-1]
        self.population.particle[0].pBest.acc = hist.history['accuracy'][-1]

        print("Current gBest acc: " + str(self.gBest_acc[0]) + "\n")
        print("Current gBest test acc: " + str(self.gBest_test_acc[0]) + "\n")

        print("Looking for a new gBest in the population...")
        for i in range(1, self.pop_size):
            print('Initialization - Particle: ' + str(i + 1))
            print(self.population.particle[i])

            self.population.particle[i].model_compile(dropout_rate)
            hist = self.population.particle[i].model_fit(self.x_train, self.y_train, batch_size=batch_size,
                                                         epochs=epochs)
            self.population.particle[i].model_delete()

            self.population.particle[i].acc = hist.history['accuracy'][-1]
            self.population.particle[i].pBest.acc = hist.history['accuracy'][-1]

            if self.population.particle[i].pBest.acc >= self.gBest_acc[0]:
                print("Found a new gBest.")
                # Use safe cloning method instead of deepcopy
                self.gBest = self.population.particle[i].clone_particle()
                self.gBest_acc[0] = self.population.particle[i].pBest.acc
                print("New gBest acc: " + str(self.gBest_acc[0]))

                self.gBest.model_compile(dropout_rate)
                test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)
                self.gBest_test_acc[0] = test_metrics[1]
                print("New gBest test acc: " + str(self.gBest_acc[0]))

            self.gBest.model_delete()

    def fit(self, Cg, dropout_rate):
        for i in range(1, self.n_iter):
            gBest_acc = self.gBest_acc[i - 1]
            gBest_test_acc = self.gBest_test_acc[i - 1]

            for j in range(self.pop_size):
                print('Iteration: ' + str(i) + ' - Particle: ' + str(j + 1))

                # Update particle velocity
                self.population.particle[j].velocity(self.gBest.layers, Cg)

                # Update particle architecture
                self.population.particle[j].update()

                print('Particle NEW architecture: ')
                print(self.population.particle[j])

                # Compute the acc in the updated particle
                self.population.particle[j].model_compile(dropout_rate)
                hist = self.population.particle[j].model_fit(self.x_train, self.y_train, batch_size=self.batch_size,
                                                             epochs=self.epochs)
                self.population.particle[j].model_delete()

                self.population.particle[j].acc = hist.history['accuracy'][-1]

                f_test = self.population.particle[j].acc
                pBest_acc = self.population.particle[j].pBest.acc

                if f_test >= pBest_acc:
                    print("Found a new pBest.")
                    print("Current acc: " + str(f_test))
                    print("Past pBest acc: " + str(pBest_acc))
                    pBest_acc = f_test
                    # Use safe cloning method instead of deepcopy
                    self.population.particle[j].pBest = self.population.particle[j].clone_particle()

                    if pBest_acc >= gBest_acc:
                        print("Found a new gBest.")
                        gBest_acc = pBest_acc
                        # Use safe cloning method instead of deepcopy
                        self.gBest = self.population.particle[j].clone_particle()

                        self.gBest.model_compile(dropout_rate)
                        hist = self.gBest.model_fit(self.x_train, self.y_train, batch_size=self.batch_size,
                                                    epochs=self.epochs)
                        test_metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test,
                                                                 batch_size=self.batch_size)
                        self.gBest.model_delete()
                        gBest_test_acc = test_metrics[1]

            self.gBest_acc[i] = gBest_acc
            self.gBest_test_acc[i] = gBest_test_acc

            print("Current gBest acc: " + str(self.gBest_acc[i]))
            print("Current gBest test acc: " + str(self.gBest_test_acc[i]))

    def fit_gBest(self, batch_size, epochs, dropout_rate):
        print("\nFurther training gBest model...")
        self.gBest.model_compile(dropout_rate)

        trainable_count = 0
        for i in range(len(self.gBest.model.trainable_weights)):
            trainable_count += K.count_params(self.gBest.model.trainable_weights[i])

        print("gBest's number of trainable parameters: " + str(trainable_count))
        self.gBest.model_fit_complete(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)

        return trainable_count

    def evaluate_gBest(self, batch_size):
        print("\nEvaluating gBest model on the test set...")

        metrics = self.gBest.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)

        print("\ngBest model loss in the test set: " + str(metrics[0]) + " - Test set accuracy: " + str(metrics[1]))
        return metrics

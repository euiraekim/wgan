import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt

import os, time

class WGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        
        self.learning_rate = 0.00005

        self.batch_size = 64
        self.BUFFER_SIZE = 60000
        self.num_examples_to_generate = 16
        self.epochs = 50

        self.n_critic = 5
        self.clip_value = 0.01

        self.G = self.build_generator()
        self.D = self.build_critic()

        self.generator_optimizer = RMSprop(self.learning_rate)
        self.discriminator_optimizer = RMSprop(self.learning_rate)

        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer, G=self.G, D=self.D)

        self.seed = tf.random.normal([self.num_examples_to_generate, self.latent_dim])

    def d_loss(self, real_output, fake_output):
        loss = tf.math.reduce_mean(real_output) - tf.math.reduce_mean(fake_output)
        return loss

    def g_loss(self, fake_output):
        loss = tf.math.reduce_mean(fake_output)
        return loss

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256 * 7 * 7, input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 256)))
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization())    
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization())    
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
        model.add(Activation('tanh'))

        model.summary()
        return model

    def build_critic(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()
        return model

    @tf.function
    def train_step(self, images):
        # critic
        for i in range(self.n_critic):
            noise = tf.random.normal([self.batch_size, self.latent_dim])
            with tf.GradientTape() as disc_tape:                
                generated_images = self.G(noise, training=False)
                real_output = self.D(images, training=True)
                fake_output = self.D(generated_images, training=True)
                disc_loss = self.d_loss(real_output, fake_output)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))
            
            for w in self.D.trainable_variables:
                w.assign(tf.clip_by_value(w, -self.clip_value, self.clip_value))

        # generator
        with tf.GradientTape() as gen_tape:
            noise = tf.random.normal([self.batch_size, self.latent_dim])

            generated_images = self.G(noise, training=True)   
            fake_output = self.D(generated_images, training=False)
            gen_loss = self.g_loss(fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))    
        
        return gen_loss, disc_loss

    def train(self):
        (train_images, _), (_, _) = mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.BUFFER_SIZE).batch(self.batch_size)

        for epoch in range(self.epochs):
            start = time.time()
            
            gen_loss_list = []
            disc_loss_list = []
            
            for image_batch in train_dataset:
                loss = self.train_step(image_batch)
                gen_loss_list.append(loss[0])
                disc_loss_list.append(loss[1])
                
            self.generate_and_save_images(self.G, epoch + 1, self.seed)
            
            if (epoch + 1) % 10 == 0:
               self.checkpoint.save(file_prefix = self.checkpoint_prefix)
        
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            print ('G_Loss is {}, D_Loss is {}'.format(sum(gen_loss_list)/len(gen_loss_list), 
                                                    sum(disc_loss_list)/len(disc_loss_list)))

        self.generate_and_save_images(self.G, self.epochs, self.seed)

    def generate_and_save_images(self, model, epoch, test_input):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 5)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    wgan = WGAN()
    wgan.train()
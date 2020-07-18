import tensorflow as tf

class BaseExperiment:

    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = {
            'train_loss':       tf.keras.metrics.Mean(name='train_loss'),
            'train_accuracy':   tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
            'test_loss':        tf.keras.metrics.Mean(name='test_loss'),
            'test_accuracy':    tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy'),
        }
        self.history = {metric: [] for metric in self.metrics}

    @tf.function
    def train_step(self, batch):
        inputs, labels = batch
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            loss = self.loss(labels, outputs) + tf.reduce_sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.metrics['train_loss'].update_state(loss)
        self.metrics['train_accuracy'].update_state(labels, outputs)

    @tf.function
    def test_step(self, batch):
        inputs, labels = batch
        outputs = self.model(inputs, training=False)
        loss = self.loss(labels, outputs) + tf.reduce_sum(self.model.losses)
        self.metrics['test_loss'].update_state(loss)
        self.metrics['test_accuracy'].update_state(labels, outputs)

    def train(self, train_dataset, test_dataset, num_epochs):
        for epoch in range(num_epochs):

            for metric in self.metrics:
                self.metrics[metric].reset_states()

            for batch in train_dataset:
                train_loss = self.train_step(batch)

            for batch in test_dataset:
                test_loss = self.test_step(batch)

            for metric in self.metrics:
                self.history[metric].append(self.metrics[metric].result())

            print(f'Epoch {epoch}')
            print(f'Train Loss={self.history["train_loss"][-1]}, Train Accuracy={self.history["train_accuracy"][-1]}')
            print(f'Test Loss={self.history["test_loss"][-1]}, Test Accuracy={self.history["test_accuracy"][-1]}')

class ImportanceWeightExperiment(BaseExperiment):
    def __init__(self, model, loss, optimizer, importance_weights):
        super().__init__(model, loss, optimizer)
        self.importance_weights = importance_weights

    @tf.function
    def train_step(self, batch):
        inputs, labels = batch
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            per_example_loss = self.loss(labels, outputs)
            per_example_weight = tf.gather(self.importance_weights, labels)
            loss = tf.tensordot(per_example_loss, per_example_weight, 1) + tf.reduce_sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.metrics['train_loss'].update_state(loss)
        self.metrics['train_accuracy'].update_state(labels, outputs)

    @tf.function
    def test_step(self, batch):
        inputs, labels = batch
        outputs = self.model(inputs, training=False)
        per_example_loss = self.loss(labels, outputs)
        loss = tf.reduce_mean(per_example_loss) + tf.reduce_sum(self.model.losses)
        self.metrics['test_loss'].update_state(loss)
        self.metrics['test_accuracy'].update_state(labels, outputs)

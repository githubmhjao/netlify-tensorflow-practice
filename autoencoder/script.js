import {MnistData} from './data.js';

async function showExamples(data) {
  // Create a container in the visor
  const surface = document.getElementById("container-surface");  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel() {
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;
  
  const encoder = tf.sequential();
  
  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.
  encoder.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  encoder.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Repeat another conv2d + maxPooling stack. 
  // Note that we have more filters in the convolution.
  encoder.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  encoder.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  encoder.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_LATENT = 10;
  encoder.add(tf.layers.dense({
    units: NUM_LATENT,
    kernelInitializer: 'varianceScaling',
    activation: 'linear'
  }));
  
  const decoder = tf.sequential();
  
  decoder.add(tf.layers.dense({
    inputShape: [NUM_LATENT,],
    units: 256,
    kernelInitializer: 'varianceScaling',
    activation: 'linear'
  }))
  
  decoder.add(tf.layers.reshape({targetShape: [4, 4, 16]}))
  
  decoder.add(tf.layers.upSampling2d({size: [2, 2]}))
  
  decoder.add(tf.layers.conv2dTranspose({
    filters: 16,
    kernelSize: 5,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }))
  
  decoder.add(tf.layers.upSampling2d({size: [2, 2]}))
  
  decoder.add(tf.layers.conv2dTranspose({
    filters: 8,
    kernelSize: 5,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }))
  
  decoder.add(tf.layers.conv2dTranspose({
    filters: 1,
    kernelSize: 1,
    strides: 1,
    activation: 'sigmoid',
    kernelInitializer: 'varianceScaling'
  }))
  
  model.add(encoder);
  model.add(decoder);
  
  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['mse'],
  });

  return model;
}

async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'mse', 'val_mse'];
  const fitCallbacks = tfvis.show.fitCallbacks(document.getElementById("container-train"), metrics);
  
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  return model.fit(trainXs, trainXs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testXs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

async function predict(model, data) {
  // Create a container in the visor
  const surface = document.getElementById("container-result");  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  const examplesPred = model.predict(examples.xs.reshape([numExamples, 28, 28, 1]))
  console.log(examplesPred)
  console.log(examplesPred.slice([0, 0, 0, 0], [1, 28, 28, 1]))
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examplesPred.slice([i, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {  
  const data = new MnistData();
  await data.load();
  await showExamples(data);
  
  const model = getModel();
  tfvis.show.modelSummary(document.getElementById('container-model'), model);
  
  await train(model, data);
  await predict(model, data);
}

document.addEventListener('DOMContentLoaded', run);

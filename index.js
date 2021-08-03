const tf = require("@tensorflow/tfjs-node");
const path = require("path");


const main = async() =>{
  const input = tf.tensor2d([
    [1, 1, 1],
    [0, 0, 0],
    [0, 1, 0],
  ]);
  const output = tf.tensor2d([[0],[1],[0]]);
  
  const layer = tf.layers.dense({ units: 10, inputShape: [3]});
  
  const outputLayer = tf.layers.dense({units: 1,inputShape: [10]});
  
  const model = tf.sequential({ layers: [layer,outputLayer] });
  
  model.compile({
    optimizer: tf.train.adam(.01),
    metrics: ["accuracy"],
    loss: "meanSquaredError",
  });
  
  const h = await model.fit(input, output,{epochs:500});

  const prediction = model.predict(tf.tensor2d([[.50,.22,.33]]));

  prediction.print();
  input.print();
  output.print();
  
  model.save("file://./model");
  
}


main()
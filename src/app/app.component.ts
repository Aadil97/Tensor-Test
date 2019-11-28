import { Component, ViewChild, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  
  linearModel: tf.Sequential;
  prediction: any;

  ngOnInit() {
    this.trainNewModel();
  }
  async trainNewModel() {
    //Defines linear regression
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));
    //sgd = stochastic gradient descent, prepares model for training
    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    //Training data, 1 dimension tensor similiar to an array
    //Data from https://www.kaggle.com/yersever/500-person-gender-height-weight-bodymassindex
    //xs weight (Kg)
    const xs = tf.tensor1d([104, 61, 104, 90, 101, 64, 95, 114, 72, 89, 70, 100, 75, 105, 96, 111, 66, 60, 108, 106]);
    //ys height (cm)
    const ys = tf.tensor1d([195, 149, 189, 174, 192, 151, 190, 197, 161, 187, 164, 185, 164, 190, 183, 194, 149, 148, 194, 194]);

    await this.linearModel.fit(xs,ys)
    console.log('model is trained!');

  }
  linearPrediction(val) {
    const value = parseInt(val);
    const output = this.linearModel.predict(tf.tensor2d([value],[1,1])) as any;
    this.prediction = Array.from(output.dataSync())[0];
  }

}

<template>
<div>
  <div id="loading">
  </div>
  <div id='main'>
    <div id="status"></div>
    <div id='results' style='display:none'>
      <div id="multi">
        <canvas />
      </div>
      <div id="single"  style='display:none'>
        <canvas />
      </div>
    </div>
    <button v-on:click="runImageSeries">Greet</button>
  </div>
</div>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';
import Util from '../services/util';

const stats = new Stats();
const imageBucket = '/static/testimages/';
const guiState = {
  outputStride: 16,
  image: 'frame5.jpg',
  singlePoseDetection: {
    minPartConfidence: 0.5,
    minPoseConfidence: 0.5,
  },
  multiPoseDetection: {
    minPartConfidence: 0.5,
    minPoseConfidence: 0.5,
    scoreThreshold: 0.5,
    nmsRadius: 20.0,
    maxDetections: 15,
  },
  showKeypoints: true,
  showSkeleton: true,
  visualizeOutputs: {
    part: 0,
    showHeatmap: true,
    showOffsets: false,
    showDisplacements: false,
  },
};
let image = null;
let modelOutputs = null;
const {
  partIds,
  poseChain
} = posenet;
/**
 * Define the skeleton by part id. This is used in multi-pose estimation. This
 *defines the parent->child relationships of our tree. Arbitrarily this defines
 *the nose as the root of the tree.
 **/
const parentChildrenTuples = poseChain.map(
  ([parentJoinName, childJoinName]) =>
  ([partIds[parentJoinName], partIds[childJoinName]]));

/**
 * Parent to child edges from the skeleton indexed by part id.  Indexes the edge
 * ids by the part ids.
 */
const parentToChildEdges =
  parentChildrenTuples.reduce((result, [partId], i) => {
    if (result[partId]) {
      result[partId] = [...result[partId], i];
    } else {
      result[partId] = [i];
    }

    return result;
  }, {});

/**
 * Child to parent edges from the skeleton indexed by part id.  Indexes the edge
 * ids by the part ids.
 */
const childToParentEdges =
  parentChildrenTuples.reduce((result, [, partId], i) => {
    if (result[partId]) {
      result[partId] = [...result[partId], i];
    } else {
      result[partId] = [i];
    }

    return result;
  }, {});

export default {
  name: 'Dance',
  mounted() {
    this.onMounted();
  },

  methods: {
    async onMounted() {
      this.net = await posenet.load();
      const net = this.net;
      this.setupGui(net);

      //await this.testImageAndEstimatePoses(net);
      document.getElementById('loading').style.display = 'none';
      document.getElementById('main').style.display = 'block';
    },

    runImageSeries() {
      console.log('agag')
      const net = this.net;
      const testImgs = this.getImages();
      const len = testImgs.length;
      let count = 0;
      const testImgFunc = this.testImageAndEstimatePoses;
      let intervalID = setInterval(function() {
        guiState.image = testImgs[count];
        count++;
        console.log('pred', testImgs[count]);
        testImgFunc(net);
        if (count === len - 1) {
          window.clearInterval(intervalID);
        }
      }, 1000);
    },
    /**
     * Converts the raw model output results into multi-pose estimation results
     */
    async decodeMultiplePosesAndDrawResults() {
      if (!modelOutputs) {
        return;
      }

      const poses = await posenet.decodeMultiplePoses(
        modelOutputs.heatmapScores, modelOutputs.offsets,
        modelOutputs.displacementFwd, modelOutputs.displacementBwd,
        guiState.outputStride, guiState.multiPoseDetection.maxDetections,
        guiState.multiPoseDetection);

      this.drawMultiplePosesResults(poses);
    },

    /**
     * Draws a pose if it passes a minimum confidence onto a canvas.
     * Only the pose's keypoints that pass a minPartConfidence are drawn.
     */
    drawResults(canvas, poses, minPartConfidence, minPoseConfidence) {
      Util.renderImageToCanvas(image, [1000, 1000], canvas);
      poses.forEach((pose) => {
        if (pose.score >= minPoseConfidence) {
          if (guiState.showKeypoints) {
            Util.drawKeypoints(
              pose.keypoints, minPartConfidence, canvas.getContext('2d'));
          }

          if (guiState.showSkeleton) {
            Util.drawSkeleton(
              pose.keypoints, minPartConfidence, canvas.getContext('2d'));
          }
        }
      });
    },

    visualizeOutputs(
      partId, drawHeatmaps, drawOffsetVectors, drawDisplacements, ctx) {
      const {
        heatmapScores,
        offsets,
        displacementFwd,
        displacementBwd
      } = modelOutputs;
      const outputStride = +guiState.outputStride;

      const [height, width] = heatmapScores.shape;

      ctx.globalAlpha = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const score = heatmapScores.get(y, x, partId);

          // to save on performance, don't draw anything with a low score.
          if (score < 0.05) continue;

          // set opacity of drawn elements based on the score
          ctx.globalAlpha = score;

          if (drawHeatmaps) {
            Util.drawPoint(ctx, y * outputStride, x * outputStride, 2, 'yellow');
          }

          const offsetsVectorY = offsets.get(y, x, partId);
          const offsetsVectorX = offsets.get(y, x, partId + 17);

          if (drawOffsetVectors) {
            Util.drawOffsetVector(
              ctx, y, x, outputStride, offsetsVectorY, offsetsVectorX);
          }

          if (Util.drawDisplacements) {
            // exponentially affect the alpha of the displacements;
            ctx.globalAlpha *= score;

            Util.drawDisplacementEdgesFrom(
              ctx, partId, displacementFwd, outputStride, parentToChildEdges, y,
              x, offsetsVectorY, offsetsVectorX);

            Util.drawDisplacementEdgesFrom(
              ctx, partId, displacementBwd, outputStride, childToParentEdges, y,
              x, offsetsVectorY, offsetsVectorX);
          }
        }

        ctx.globalAlpha = 1;
      }
    },

    drawMultiplePosesResults(poses) {
      const canvas = this.multiPersonCanvas();
      this.drawResults(
        canvas, poses, guiState.multiPoseDetection.minPartConfidence,
        guiState.multiPoseDetection.minPoseConfidence);

      const {
        part,
        showHeatmap,
        showOffsets,
        showDisplacements
      } =
      guiState.visualizeOutputs;
      const partId = +part;

      this.visualizeOutputs(
        partId, showHeatmap, showOffsets, showDisplacements,
        canvas.getContext('2d'));
    },

    async testImageAndEstimatePoses(net) {
      //this.setStatusText('Predicting...');
      //document.getElementById('results').style.display = 'none';

      // Purge prevoius variables and free up GPU memory
      this.disposeModelOutputs();

      // Load an example image
      image = await this.loadImage(guiState.image);

      // Creates a tensor from an image
      const input = tf.fromPixels(image);

      // Stores the raw model outputs from both single- and multi-pose results can
      // be decoded.
      // Normally you would call estimateSinglePose or estimateMultiplePoses,
      // but by calling this method we can previous the outputs of the model and
      // visualize them.
      modelOutputs = await net.predictForMultiPose(input, guiState.outputStride);

      // Process the model outputs to convert into poses
      await this.decodeSingleAndMultiplePoses();

      //this.setStatusText('');
      document.getElementById('results').style.display = 'block';
      //input.dispose();
    },

    async loadImage(imagePath) {
      const image = new Image();
      const promise = new Promise((resolve, reject) => {
        image.crossOrigin = '';
        image.onload = () => {
          resolve(image);
        };
      });

      image.src = `${imageBucket}${imagePath}`;
      return promise;
    },

    setStatusText(text) {
      const resultElement = document.getElementById('status');
      resultElement.innerText = text;
    },

    async decodeSinglePoseAndDrawResults() {
      if (!modelOutputs) {
        return;
      }

      const pose = await posenet.decodeSinglePose(
        modelOutputs.heatmapScores, modelOutputs.offsets, guiState.outputStride);

      this.drawSinglePoseResults(pose);
    },

    singlePersonCanvas() {
      return document.querySelector('#single canvas');
    },

    multiPersonCanvas() {
      return document.querySelector('#multi canvas');
    },
    /**
     * Draw the results from the single-pose estimation on to a canvas
     */
    drawSinglePoseResults(pose) {
      const canvas = this.singlePersonCanvas();
      this.drawResults(
        canvas, [pose], guiState.singlePoseDetection.minPartConfidence,
        guiState.singlePoseDetection.minPoseConfidence);

      const {
        part,
        showHeatmap,
        showOffsets
      } = guiState.visualizeOutputs;
      // displacements not used for single pose decoding
      const showDisplacements = false;
      const partId = +part;

      this.visualizeOutputs(
        partId, showHeatmap, showOffsets, showDisplacements,
        canvas.getContext('2d'));
    },

    decodeSingleAndMultiplePoses() {
      //this.decodeSinglePoseAndDrawResults();
      this.decodeMultiplePosesAndDrawResults();
    },

    /**
     * Purges variables and frees up GPU memory using dispose() method
     */
    disposeModelOutputs() {
      if (modelOutputs) {
        modelOutputs.heatmapScores.dispose();
        modelOutputs.offsets.dispose();
        modelOutputs.displacementFwd.dispose();
        modelOutputs.displacementBwd.dispose();
      }
    },

    getImages() {
      let testImgs = [];
      for (let i = 5; i < 290; i++) {
        testImgs.push(`frame${i}.jpg`);
      }
      return testImgs;
    },

    setupGui(net) {
      //const gui = new dat.GUI();
      // Pose confidence: the overall confidence in the estimation of a person's
      // pose (i.e. a person detected in a frame)
      // Min part confidence: the confidence that a particular estimated keypoint
      // position is accurate (i.e. the elbow's position)

    }
  }
}
</script>

import * as tf from '@tensorflow/tfjs'
import * as posenet from '@tensorflow-models/posenet';

export default {
  getName() {
    return '';
  },

  color: 'aqua',
  lineWidth: 2,

  toTuple(position) {
    return [position.y, position.x];
  },

  drawPoint(ctx, y, x, r) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = 'pink';
    ctx.fill();
  },

  drawSegment(pa, pb, color, scale, ctx) {
    ctx.beginPath();
    ctx.moveTo(pa[1] * scale, pa[0] * scale);
    ctx.lineTo(pb[1] * scale, pb[0] * scale);
    ctx.lineWidth = 5;
    ctx.strokeStyle = color;
    ctx.stroke();
  },

  drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
    const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);
    adjacentKeyPoints.forEach((keypoints) => {
      this.drawSegment(this.toTuple(keypoints[0].position),
        this.toTuple(keypoints[1].position), 'pink', scale, ctx);
    });
  },

  /**
   * Draw pose keypoints onto a canvas
   */
  drawKeypoints(keypoints, minConfidence, ctx, color, scale = 1) {
    for (let i = 0; i < keypoints.length; i++) {
      const keypoint = keypoints[i];

      if (keypoint.score < minConfidence) {
        continue;
      }

      const {
        y,
        x
      } = keypoint.position;
      this.drawPoint(ctx, y * scale, x * scale, 2, color);
    }
  },

  /**
   * Draw the bounding box of a pose. For example, for a whole person standing
   * in an image, the bounding box will begin at the nose and extend to one of
   * ankles
   */
  drawBoundingBox(keypoints, ctx) {
    const boundingBox = posenet.getBoundingBox(keypoints);

    ctx.rect(boundingBox.minX, boundingBox.minY,
      boundingBox.maxX - boundingBox.minX, boundingBox.maxY - boundingBox.minY);

    ctx.stroke();
  },

  /**
   * Converts an arary of pixel data into an ImageData object
   */
  async renderToCanvas(a, ctx) {
    const [height, width] = a.shape;
    const imageData = new ImageData(width, height);

    const data = await a.data();

    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      const k = i * 3;

      imageData.data[j] = data[k];
      imageData.data[j + 1] = data[k + 1];
      imageData.data[j + 2] = data[k + 2];
      imageData.data[j + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
  },

  /**
   * Draw an image on a canvas
   */
  renderImageToCanvas(image, size, canvas) {
    canvas.width = size[0];
    canvas.height = size[1];
    const ctx = canvas.getContext('2d');

    //ctx.drawImage(image, 0, 0);
  },

  /**
   * Draw heatmap values, one of the model outputs, on to the canvas
   * Read our blog post for a description of PoseNet's heatmap outputs
   * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
   */
  drawHeatMapValues(heatMapValues, outputStride, canvas, color) {
    const ctx = canvas.getContext('2d');
    const radius = 5;
    const scaledValues = heatMapValues.mul(tf.scalar(outputStride, 'int32'));

    this.drawPoints(ctx, scaledValues, radius, color);
  },

  /**
   * Used by the drawHeatMapValues method to draw heatmap points on to
   * the canvas
   */
  drawPoints(ctx, points, radius, color) {
    const data = points.buffer().values;

    for (let i = 0; i < data.length; i += 2) {
      const pointY = data[i];
      const pointX = data[i + 1];

      if (pointX !== 0 && pointY !== 0) {
        ctx.beginPath();
        ctx.arc(pointX, pointY, radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
  },

  /**
   * Draw offset vector values, one of the model outputs, on to the canvas
   * Read our blog post for a description of PoseNet's offset vector outputs
   * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
   */
  drawOffsetVectors(heatMapValues, offsets, outputStride, scale = 1, ctx, color) {
    const offsetPoints = posenet.singlePose.getOffsetPoints(heatMapValues, outputStride, offsets);
    const heatmapData = heatMapValues.buffer().values;
    const offsetPointsData = offsetPoints.buffer().values;

    for (let i = 0; i < heatmapData.length; i += 2) {
      const heatmapY = heatmapData[i] * outputStride;
      const heatmapX = heatmapData[i + 1] * outputStride;
      const offsetPointY = offsetPointsData[i];
      const offsetPointX = offsetPointsData[i + 1];

      this.drawSegment([heatmapY, heatmapX], [offsetPointY, offsetPointX], color, scale, ctx);
    }
  },


  drawOffsetVector(
    ctx, y, x, outputStride, offsetsVectorY, offsetsVectorX) {
    this.drawSegment(
      [y * outputStride, x * outputStride], [y * outputStride + offsetsVectorY, x * outputStride + offsetsVectorX],
      'red', 1.0, ctx);
  },

  drawDisplacementEdgesFrom(
    ctx, partId, displacements, outputStride, edges, y, x, offsetsVectorY,
    offsetsVectorX) {
    const numEdges = displacements.shape[2] / 2;

    const offsetX = x * outputStride + offsetsVectorX;
    const offsetY = y * outputStride + offsetsVectorY;

    const edgeIds = edges[partId] || [];

    if (edgeIds.length > 0) {
      edgeIds.forEach((edgeId) => {
        const displacementY = displacements.get(y, x, edgeId);
        const displacementX = displacements.get(y, x, edgeId + numEdges);

        this.drawSegment(
          [offsetY, offsetX], [offsetY + displacementY, offsetX + displacementX], 'blue', 1.0, ctx);
      });
    }
  }
}

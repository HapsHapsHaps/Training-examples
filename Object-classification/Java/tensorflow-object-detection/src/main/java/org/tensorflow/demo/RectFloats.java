package org.tensorflow.demo;

public class RectFloats {
    private final float left, top, right, bottom;

    public RectFloats(RectFloats rectFloats) {
        this.left = rectFloats.getLeft();
        this.top = rectFloats.getTop();
        this.right = rectFloats.getRight();
        this. bottom = rectFloats.getBottom();
    }

    public RectFloats(float left, float top, float right, float bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this. bottom = bottom;
    }

    public float getLeft() {
        return left;
    }

    public float getTop() {
        return top;
    }

    public float getRight() {
        return right;
    }

    public float getBottom() {
        return bottom;
    }
}

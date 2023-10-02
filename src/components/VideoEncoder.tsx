import React, { useRef, useState } from 'react';
import * as tf from "@tensorflow/tfjs"
import KMeans from 'tf-kmeans';

export default function VideoEncoder(props : { path : string, model : any}) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const outCanvRef = useRef(null);
    const outMapRef = useRef(null);
    const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout>()
    const [pipelineRunning, setPipelineRunning] = useState(false)
    const [lastTime, setLastTime] = useState(Date.now())
    const [fps, setFps] = useState(0);
    const [framenumber, setFramenumber] = useState(1);

    const extractFrames = async () => {
        console.log("extracting the frames");
        setPipelineRunning(true);
        setFramenumber(framenumber + 1);

        // set the fps
        const newTime = Date.now();
        setFps(1000/(newTime - lastTime));
        setLastTime(newTime);

        // set up all the references
        const video = videoRef.current;
        const canvas: any = canvasRef.current;
        const outcanvas :any = outCanvRef.current;
        const outMap : any = outMapRef.current;

        if (!video || !canvas || !outcanvas || !outMap) return;

        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const pred = await tf.tidy(()=>{
            // You can do something with the extracted frame here, like displaying it, saving it, etc.
            const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

            // Convert the image data to a TensorFlow tensor
            const inputTensor = tf.browser.fromPixels(imageData).toFloat();
    
            // Preprocess the input tensor if needed (e.g., normalize, resize)
            let normalizedInput : any = inputTensor.div(tf.scalar(127.5)).sub(tf.scalar(1));
            normalizedInput = normalizedInput.reshape([1,256,256,3]);
            // const contrastFactor = 7;

            // // Compute the mean pixel value of the image
            // const mean = tf.mean(normalizedInput);

            // // Adjust the contrast by scaling pixel values relative to the mean
            // const adjustedImage = tf.add(tf.mul(normalizedInput.sub(mean), contrastFactor), mean);

            // // Clip pixel values to be in the range [0, 255] (assuming 8-bit images)
            // const in_image_clean = tf.clipByValue(adjustedImage, 0, 1);
    
            // Run the model prediction
            const dehazedPrediction = props.dehazeModel.predict(normalizedInput);
            // const outputTensor = props.segModel.predict(normalizedInput);

            // // // create a mask and get the correct mapping
            // const argmaxMask = tf.argMax(outputTensor, -1);

            // // // Expand dimensions by adding a new axis at position 0
            // const expandedMask = argmaxMask.reshape([256,256,1]);

            // // creating therequired mapping
            const colorMap = [
                [255, 0, 0],      // Red
                [0, 255, 0],      // Green
                [0, 0, 255],      // Blue
                [255, 255, 0],    // Yellow
                [255, 0, 255],    // Magenta
                [0, 255, 255],    // Cyan
                [128, 0, 0],      // Maroon
                [0, 128, 0],      // Green (dark)
                [0, 0, 128],      // Navy
                [128, 128, 0],    // Olive
                [128, 0, 128],    // Purple
                [0, 128, 128],    // Teal
                [192, 192, 192],  // Silver
                [128, 128, 128],  // Gray
                [255, 165, 0],    // Orange
                [210, 180, 140],  // Tan
                [255, 192, 203],  // Pink
                [0, 128, 128],    // Aqua
                [0, 0, 0],        // Black
                [255, 255, 255],  // White
                [255, 99, 71],    // Tomato
                [0, 139, 139],    // Dark Cyan
                [255, 20, 147],   // Deep Pink
                [0, 255, 127]     // Spring Green
            ];              

            // // Create a new tensor with the RGB values from the color map
            // const colorTensor = tf.tensor(colorMap, [24, 3]);

            // // Gather colors based on input values
            // const mappedColors = tf.gather(colorTensor, expandedMask.flatten(), 0).asType("int32");

            // const outImage : tf.Tensor3D = mappedColors.reshape([256, 256, 3]);

            // Reshape the tensor to a 256,256,3
            normalizedInput = dehazedPrediction.reshape([256,256,3]);

            // using a k-means model:
            const predictions = kmeans.Train(normalizedInput.reshape([256*256,3])).reshape([256,256,1]);
            const imageSegMap = tf.gather(colorMap, predictions).reshape([256,256,3]);
            console.log("The predictions shape is ", imageSegMap.shape);

            // converting to an image
            const outImage : tf.Tensor3D = dehazedPrediction.reshape([256,256,3]).add(tf.scalar(1)).mul(tf.scalar(127.5)).cast('int32');
            const overlayMap : tf.Tensor3D = imageSegMap.cast("int32");

            if(outCanvRef.current) {
                tf.browser.toPixels(outImage, outCanvRef.current);
                tf.browser.toPixels(overlayMap, outMap);
            } else {
                console.log("The pixels could not be printed to the canvas");
            }
        })

        // Continue extracting frames (e.g., every 10ms)
        const handle = setTimeout(extractFrames, 10);
        setTimeoutId(handle);
    };

    const stopExtraction = () => {
        console.log("stopping the extraction")
        setPipelineRunning(false);
        clearTimeout(timeoutId);
    }
    
    return (
        <div className=' font-display'>
            <div className={" p-2 m-2 rounded-md bg-green-600 animate-pulse"}>
                <span>Processing Frame {framenumber}</span>
            </div>
            <div className="grid grid-rows-2 gap-2 justify-center content-center w-full my-2">
                <video
                    className='hidden'
                    ref={videoRef}
                    width={256}
                    height={256}
                    controls
                    onLoadedData={extractFrames}
                    autoPlay = {true}
                    loop = {true}
                >
                    <source src={props.path} type="video/mp4" />
                </video>
                <div className=" w-full bg-neutral-600 rounded-md overflow-hidden">
                    <div className="px-2">
                        Input
                    </div>
                    <canvas ref={canvasRef} width={256} height={256} style={{ display: 'block', background: "white" }} className=' rounded-md   '/>
                </div>
                <div className="w-full bg-neutral-600 rounded-md overflow-hidden relative">  
                    <div className=" relative">
                        <canvas ref={outCanvRef} width={256} height={256} style={{ display: 'block', background: "white" }} className=' absolute top-0 left-0 z-0 rounded-md'/>
                        <canvas ref={outMapRef} width={256} height={256} style={{ display: 'block' }} className=' absolute top-0 left-0 opacity-25 z-10 rounded-md'/>
                    </div>
                    <div className=" top-[256px] absolute px-2">
                        Output
                    </div>
                </div>
            </div>
            <div className=" p-2 grid grid-cols-3 gap-1">
                <button onClick={extractFrames} className={' p-2 bg-green-600 rounded-md'} disabled={pipelineRunning}>Start</button>
                <button onClick={stopExtraction} className=' p-2 bg-red-600 rounded-md'>Stop</button>
                <div className=' p-2 bg-blue-600 text-center rounded-md'>FPS: {fps.toFixed(4)}</div>

            </div>
        </div>
    );
}
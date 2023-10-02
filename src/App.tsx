import { useState, useRef, useEffect } from 'react'
import VideoEncoder from './components/VideoEncoder'
import * as tf from '@tensorflow/tfjs';
import './App.css'

function App() {
  const [path, setPath] = useState("")
  const unprocessed = useRef(null)
  const processed = useRef(null)

  const getTempFilePath = (evt : any) => {
    const file = evt.target.files[0]
    const localPath = URL.createObjectURL(file);
    console.log('Local path:', localPath);
    setPath(localPath);
  }
  const [model, setModel] = useState<tf.LayersModel>();

  useEffect(() => {
    const loadModel = async () => {
      try {
        const _model = await tf.loadLayersModel("/models/model/model.json");
        setModel(_model);
      } catch (error) {
        console.error('Error loading the model:', error);
      }
    };

    loadModel();
  }, []);

  return (
    <div className="App md:px-[30%] min-h-screen">
      {model? "": <div className="p-2 font-display">Loading Model...</div>}
      <div className=" text-3xl p-2 font-bold text-center bg-neutral-600 rounded-md mx-2 font-display">FOG VISION</div>
      {path==""? 
        <div
          className='p-2 font-display'
        >
          <input type="file" name="file-in" id="file-in" accept='.mp4' onChange={getTempFilePath} className=' hidden'/>
          <label htmlFor="file-in" className=' p-2 rounded-md h-[30vh] flex justify-center items-center w-full bg-slate-600 mb-2 text-lg'>Click to upload custom test video</label>
          <button
            className='p-2 rounded-md h-[30vh] flex justify-center items-center w-full bg-slate-600 text=lg'
            onClick={()=>{
              setPath("/GT_04_01.mp4")
            }}
          >Use Default Test Video</button>
        </div>:
        <VideoEncoder path={path} model={model}/>
      }
    </div>
  )
}

export default App

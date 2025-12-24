import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Mic, MicOff, Upload, X, ArrowLeft, Check, Play, Pause } from 'lucide-react';
import { uploadToCloudinary } from '../cloudinary/config';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '../firebase/config';

const AudioTest = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  
  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);
  const { currentUser } = useAuth();
  const navigate = useNavigate();

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      const chunks = [];
      mediaRecorder.ondataavailable = (event) => {
        chunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setAudioBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        setError('');
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      setError('Microphone access denied. Please allow microphone access and try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type.startsWith('audio/')) {
        setSelectedFile(file);
        setError('');
        
        // Create preview
        const url = URL.createObjectURL(file);
        setPreview(url);
      } else {
        setError('Please select a valid audio file');
      }
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

 const handleSubmit = async () => {
  const fileToUpload = selectedFile || audioBlob;

  if (!fileToUpload) {
    setError("Please record audio or select a file first");
    return;
  }

  setUploading(true);
  setError("");

  try {
    const formData = new FormData();

    if (fileToUpload instanceof Blob && !fileToUpload.name) {
      formData.append("file", fileToUpload, "recorded_audio.wav");
    } else {
      formData.append("file", fileToUpload);
    }

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (response.ok) {
      const {result, confidence, isAffected } = data;

      navigate("/results", {
        state: {
          testType: "audio",
          confidence,
          isAffected,
          fileUrl: null
        }
      });
    } else {
      throw new Error(data.error || "Prediction failed");
    }
  } catch (error) {
    console.error("Prediction error:", error);
    setError("Failed to analyze audio. Please try again.");
  } finally {
    setUploading(false);
  }
};


  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-gray-600 hover:text-blue-600 transition-colors duration-200 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Home
          </button>
          <h1 className="text-3xl font-bold text-gray-800">Audio Test</h1>
          <p className="text-gray-600 mt-2">
            Record or upload your voice for vocal pattern analysis and potential Parkinson's detection.
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-8">
          
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Upload Audio File</h2>
            
            {!selectedFile ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-400 hover:bg-blue-50 transition-colors duration-200 cursor-pointer"
              >
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg text-gray-600 mb-2">Click to upload audio file</p>
                <p className="text-sm text-gray-500">MP3, WAV, M4A up to 50MB</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-center gap-4">
                  <button
                    onClick={togglePlayback}
                    className="bg-blue-500 text-white px-6 py-3 rounded-full hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200 flex items-center gap-2"
                  >
                    {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                    {isPlaying ? 'Pause' : 'Play'} File
                  </button>
                  <button
                    onClick={handleRemoveFile}
                    className="bg-red-500 text-white px-6 py-3 rounded-full hover:bg-red-600 focus:ring-2 focus:ring-red-500 focus:ring-offset-2 transition-colors duration-200 flex items-center gap-2"
                  >
                    <X className="w-5 h-5" />
                    Remove
                  </button>
                </div>
                <p className="text-center text-gray-600">
                  Selected: {selectedFile.name}
                </p>
                <audio
                  ref={audioRef}
                  src={preview}
                  onEnded={() => setIsPlaying(false)}
                  className="hidden"
                />
              </div>
            )}

            {error && (
              <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}
          </div>

          {/* Instructions */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Instructions</h3>
            <div className="bg-blue-50 rounded-lg p-6">
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start gap-3">
                  <Check className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                  <span>Speak clearly and at a normal pace</span>
                </li>
                <li className="flex items-start gap-3">
                  <Check className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                  <span>Record for at least 10-15 seconds</span>
                </li>
                <li className="flex items-start gap-3">
                  <Check className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                  <span>Ensure a quiet environment with minimal background noise</span>
                </li>
                <li className="flex items-start gap-3">
                  <Check className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                  <span>You can read a passage or speak naturally about your day</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-center">
            <button
              onClick={handleSubmit}
              disabled={(!audioBlob && !selectedFile) || uploading}
              className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-medium hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 flex items-center gap-2"
            >
              {uploading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Check className="w-5 h-5" />
                  Submit for Analysis
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AudioTest;

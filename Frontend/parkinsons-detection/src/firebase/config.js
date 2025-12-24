import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  // Replace with your Firebase config
  apiKey: "AIzaSyBSl0vGkTqujuro_DPi1wdtksyBve-WH0E",
  authDomain: "parkinsons-detection-22942.firebaseapp.com",
  projectId: "parkinsons-detection-22942",
  storageBucket: "parkinsons-detection-22942.firebasestorage.app",
  messagingSenderId: "161423760405",
  appId: "1:161423760405:web:a47d1da1382a36275d021c",
  measurementId: "G-YN1E64VKNL"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export default app;


import { Link } from 'react-router-dom';
import { useRef } from 'react';
import { ArrowRight, Heart } from 'lucide-react';

const Home = () => {
  const testSectionRef = useRef(null);
  const handleScroll = () => {
    testSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Hero Section */}
      <div className="bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Left side - Text Content */}
            <div className="space-y-8">
              <h1 className="text-4xl lg:text-5xl font-bold text-gray-800 leading-tight">
                Early Detection,<br />
                Better Outcomes
              </h1>
              <p className="text-lg text-gray-600 leading-relaxed">
                Our innovative platform helps in the early detection of Parkinson's disease 
                through simple drawing and audio tests, empowering you with timely insights and support.
              </p>
              <button onClick={handleScroll} className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-medium hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200 flex items-center gap-2">
                Get Started
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>

            {/* Right side - Doctor Illustration */}
            <div className="flex justify-center">
              <img src="/images/doctor-illustration.png" alt="Doctor" />
            </div>
          </div>
        </div>
      </div>

      {/* Test Selection Section */}
      <div ref={testSectionRef} className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">
            Choose Your Detection Method
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Drawing Test Card */}
          <div className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-xl transition-shadow duration-300">
            <div className="text-center mb-6">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Drawing Test</h3>
              <p className="text-gray-600">Upload a drawing for analysis of motor function.</p>
            </div>

            <div className="mb-6 rounded-lg">
              <img src="/images/spiral-drawing.png" alt="Drawing" className='w-full h-52 rounded-lg' />
            </div>

            <p className="text-gray-600 mb-6">
              This test analyzes subtle irregularities in your drawing patterns to identify 
              potential early indicators of Parkinson's disease.
            </p>

            <Link
              to="/drawing-test"
              className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200 flex items-center justify-center gap-2"
            >
              Start Drawing Test
            </Link>
          </div>

          {/* Audio Test Card */}
          <div className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-xl transition-shadow duration-300">
            <div className="text-center mb-6">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Audio Test</h3>
              <p className="text-gray-600">Record or upload your voice for vocal pattern analysis.</p>
            </div>

            <div className="mb-6 rounded-lg">
              <img src="/images/voice-recording.png" alt="Voice Analysis" className='w-full h-52 rounded-lg' />
            </div>

            <p className="text-gray-600 mb-6">
              Your vocal characteristics, such as pitch and tremor, are analyzed to detect 
              subtle changes often associated with early-stage Parkinson's.
            </p>

            <Link
              to="/audio-test"
              className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200 flex items-center justify-center gap-2"
            >
              Start Audio Test
            </Link>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-600 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div className="flex space-x-8">
              <a href="#" className="text-gray-300 hover:text-white transition-colors duration-200">
                Quick Links
              </a>
              <a href="#" className="text-gray-300 hover:text-white transition-colors duration-200">
                Resources
              </a>
            </div>
            <div className="flex space-x-4">
              <a href="#" className="text-gray-300 hover:text-white transition-colors duration-200">
                <img src="/images/twitter-icon.png" alt="Twitter" className="w-8 h-8" />
              </a>
              <a href="#" className="text-gray-300 hover:text-white transition-colors duration-200">
                <img src="/images/instagram-icon.png" alt="Instagram" className="w-8 h-8" />
              </a>
              <a href="#" className="text-gray-300 hover:text-white transition-colors duration-200">
                <img src="/images/linkedin-icon.png" alt="LinkedIn" className="w-8 h-8" />
              </a>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-white">Made with</span>
              <div className="w-6 h-6 items-center justify-center">
                <span className="text-white font-bold text-sm"><Heart /></span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;

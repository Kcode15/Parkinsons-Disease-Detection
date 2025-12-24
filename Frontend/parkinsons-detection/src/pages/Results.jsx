import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { collection, query, where, orderBy, getDocs } from 'firebase/firestore';
import { db } from '../firebase/config';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ArrowLeft, Brain, Mic, PenTool, Calendar, TrendingUp } from 'lucide-react';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [historicalData, setHistoricalData] = useState([]);
  const [loading, setLoading] = useState(true);

  const { testType, confidence, isAffected, fileUrl } = location.state || {};

  useEffect(() => {
    const fetchHistoricalData = async () => {
      if (!currentUser) return;

      try {
        const q = query(
          collection(db, 'testResults'),
          where('userId', '==', currentUser.uid),
          where('testType', '==', testType),
          orderBy('timestamp', 'asc')
        );
        
        const querySnapshot = await getDocs(q);
        const data = querySnapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data(),
          date: doc.data().timestamp?.toDate()?.toLocaleDateString() || 'Unknown'
        }));
        
        setHistoricalData(data);
      } catch (error) {
        console.error('Error fetching historical data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchHistoricalData();
  }, [currentUser, testType]);

  if (!location.state) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">No Results Found</h1>
          <p className="text-gray-600 mb-6">Please take a test first to see your results.</p>
          <button
            onClick={() => navigate('/')}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors duration-200"
          >
            Go Home
          </button>
        </div>
      </div>
    );
  }

  const getTestIcon = () => {
    return testType === 'drawing' ? <PenTool className="w-6 h-6" /> : <Mic className="w-6 h-6" />;
  };

  const getTestTitle = () => {
    return testType === 'drawing' ? 'Drawing Test Result' : 'Audio Test Result';
  };

  const getTestDescription = () => {
    return testType === 'drawing' 
      ? 'Comprehensive analysis of your drawing pattern.'
      : 'Comprehensive analysis of your voice recording.';
  };

  const formatChartData = () => {
    return historicalData.map((item, index) => ({
      name: `Test ${index + 1}`,
      confidence: Math.round(item.confidence),
      date: item.date
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-gray-600 hover:text-blue-600 transition-colors duration-200 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Home
          </button>
        </div>

        {/* Main Result Card */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
          <div className="flex items-center gap-3 mb-4">
            {getTestIcon()}
            <h1 className="text-2xl font-bold text-gray-800">{getTestTitle()}</h1>
          </div>
          <p className="text-gray-600 mb-6">{getTestDescription()}</p>

          {/* Status and Confidence */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-gray-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Status</h3>
              <div className="flex items-center gap-3">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  isAffected 
                    ? 'bg-pink-100 text-pink-800' 
                    : 'bg-green-100 text-green-800'
                }`}>
                  {isAffected ? 'Parkinson\'s Affected' : 'Healthy'}
                </span>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Confidence Score</h3>
              <div className="flex items-center gap-3">
                <span className="text-3xl font-bold text-blue-600">
                  {Math.round(confidence)}%
                </span>
              </div>
            </div>
          </div>

          {/* Explanation */}
          <div className="bg-blue-50 rounded-lg p-6 mb-8">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Analysis Results</h3>
            <p className="text-gray-700 leading-relaxed">
              {isAffected 
                ? `The analysis of your ${testType === 'drawing' ? 'drawing pattern' : 'voice recording'} indicates a higher likelihood of Parkinson's disease. We recommend consulting with a medical professional for further evaluation and diagnosis. Early detection is key for effective management.`
                : `The analysis of your ${testType === 'drawing' ? 'drawing pattern' : 'voice recording'} shows no significant indicators of Parkinson's disease. However, if you have concerns, we recommend consulting with a medical professional for a comprehensive evaluation.`
              }
            </p>
          </div>

          {/* Next Steps */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Next Steps & Recommendations</h3>
            <ul className="space-y-3 text-gray-700">
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                <span>Schedule an appointment with a neurologist for a professional diagnosis.</span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                <span>Share these results with your primary care physician for guidance.</span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                <span>
                  Learn more about Parkinson's disease on our{' '}
                  <button
                    onClick={() => navigate('/about')}
                    className="text-blue-600 hover:underline"
                  >
                    About the Disease
                  </button>{' '}
                  page.
                </span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                <span>Consider re-taking the test at a later date for updated results.</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Historical Confidence Trend */}
        {historicalData.length > 0 && (
          <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
            <div className="flex items-center gap-3 mb-6">
              <TrendingUp className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-bold text-gray-800">Historical Confidence Trend ({testType === 'drawing' ? 'Drawing' : 'Audio'} Tests)</h2>
            </div>
            <p className="text-gray-600 mb-6">Track your prediction confidence levels over time.</p>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={formatChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="name" 
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis 
                    domain={[0, 100]}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip 
                    formatter={(value) => [`${value}%`, 'Confidence Level']}
                    labelFormatter={(label) => `Test: ${label}`}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="confidence" 
                    stroke="#2563EB" 
                    strokeWidth={3}
                    dot={{ fill: '#2563EB', strokeWidth: 2, r: 6 }}
                    activeDot={{ r: 8, stroke: '#2563EB', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={() => navigate('/')}
            className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-medium hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200"
          >
            Take Another Test
          </button>
          <button
            onClick={() => navigate('/about')}
            className="bg-gray-600 text-white px-8 py-4 rounded-lg text-lg font-medium hover:bg-gray-700 focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors duration-200"
          >
            Learn About Parkinson's
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;

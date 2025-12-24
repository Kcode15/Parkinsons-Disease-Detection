import { Brain, Heart, Users, BookOpen, MessageCircle } from 'lucide-react';

const About = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">Understanding Parkinson's Disease</h1>
        </div>

        {/* Introduction */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-12">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">What is Parkinson's Disease?</h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                Parkinson's disease is a progressive neurological disorder that affects movement. 
                It occurs when nerve cells (neurons) in the brain that produce dopamine, a chemical 
                that helps coordinate muscle movement, begin to die or become impaired.
              </p>
              <p className="text-gray-700 leading-relaxed">
                The disease typically develops slowly over time, and symptoms often begin with 
                subtle changes that may go unnoticed. Early detection and treatment can help 
                manage symptoms and improve quality of life.
              </p>
            </div>
            <div className="flex justify-center">
              <img src="/images/brain-illustration-2.png" alt="Brain" className='w-84 h-80 rounded-2xl'/>
            </div>
          </div>
        </div>

        {/* Common Symptoms */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-800 text-center mb-8">Common Symptoms</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="w-12 h-12 bg-blue-100 rounded-md flex items-center justify-center mb-4">
                <img src="/images/tremor.png" alt="Tremor" className='w-8 h-8' />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Tremor</h3>
              <p className="text-gray-600 text-sm">
                Involuntary shaking, usually starting in the hands or fingers, often occurring at rest.
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <img src="/images/bradykinesia.png" alt="Bradykinesia" className='w-8 h-8' />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Bradykinesia</h3>
              <p className="text-gray-600 text-sm">
                Slowness of movement, making simple tasks difficult and time-consuming.
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <img src="/images/rigidity.png" alt="Rigidity" className='w-8 h-8' />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Rigidity</h3>
              <p className="text-gray-600 text-sm">
                Muscle stiffness that can occur in any part of the body, limiting range of motion.
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <img src="/images/postural-instability.png" alt="Postural Instability" className='w-8 h-8' />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Postural Instability</h3>
              <p className="text-gray-600 text-sm">
                Impaired balance and coordination, leading to difficulty with posture and walking.
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <img src="/images/speech_changes.png" alt="Speech Changes" className='w-8 h-8' />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Speech Changes</h3>
              <p className="text-gray-600 text-sm">
                Soft, quick, slurred, or hesitant speech, often with reduced facial expressions.
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <img src="/images/non-motor-symptoms.png" alt="Non-motor-symptoms" className='w-8 h-8' />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Non-Motor Symptoms</h3>
              <p className="text-gray-600 text-sm">
                Depression, anxiety, sleep problems, cognitive changes, and loss of smell.
              </p>
            </div>
          </div>
        </div>

        {/* Understanding the Causes */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-12">
          <h2 className="text-3xl font-bold text-gray-800 text-center mb-8">Understanding the Causes</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <p className="text-gray-700 leading-relaxed mb-4">
                The exact cause of Parkinson's disease is unknown, but several factors are believed to play a role:
              </p>
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                  <span><strong>Genetic factors:</strong> Certain genetic mutations can increase the risk of developing Parkinson's.</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                  <span><strong>Environmental factors:</strong> Exposure to certain toxins and chemicals may contribute to the disease.</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                  <span><strong>Age:</strong> The risk increases with age, typically affecting people over 60.</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                  <span><strong>Gender:</strong> Men are slightly more likely to develop Parkinson's than women.</span>
                </li>
              </ul>
            </div>
            <div className="flex justify-center">
              <img src="/images/dna-illustration.png" alt="Neural Network" className='w-84 h-80 rounded-2xl'/>
            </div>
          </div>
        </div>

        {/* Treatment Approaches */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-800 text-center mb-8">Treatment Approaches</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <img src="/images/tablets-illustration.png" alt="Medication" className='w-84 h-80 rounded-2xl'/>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Medication</h3>
              <p className="text-gray-600 text-sm">
                Levodopa and other medications can help manage symptoms by increasing dopamine levels in the brain.
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <img src="/images/brain-illustration-3.png" alt="Deep Brain Stimulation" className='w-84 h-80 rounded-2xl'/>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Deep Brain Stimulation (DBS)</h3>
              <p className="text-gray-600 text-sm">
                A surgical procedure that implants electrodes in the brain to help control movement symptoms.
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <img src="/images/therapy-image.png" alt="Therapies & Rehabilitation" className='w-84 h-80 rounded-2xl'/>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Therapies & Rehabilitation</h3>
              <p className="text-gray-600 text-sm">
                Physical therapy, occupational therapy, and speech therapy can help maintain function and improve quality of life.
              </p>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <img src="/images/exercise-image.png" alt="Exercise" className='w-84 h-80 rounded-2xl'/>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Lifestyle Adjustments</h3>
              <p className="text-gray-600 text-sm">
                Regular exercise, a healthy diet, and stress management can help manage symptoms and improve overall well-being.
              </p>
            </div>
          </div>
        </div>

        {/* Additional Resources */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <h2 className="text-3xl font-bold text-gray-800 text-center mb-8">Additional Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 rounded-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <BookOpen className="w-6 h-6 text-blue-600" />
                <h3 className="text-lg font-semibold text-gray-800">Frequently Asked Questions</h3>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                Find answers to common questions about our testing methods and results.
              </p>
              <button className="text-blue-600 hover:underline text-sm font-medium">
                Learn More →
              </button>
            </div>

            <div className="bg-gray-50 rounded-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <Users className="w-6 h-6 text-blue-600" />
                <h3 className="text-lg font-semibold text-gray-800">Support Communities</h3>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                Connect with others who understand and share experiences.
              </p>
              <button className="text-blue-600 hover:underline text-sm font-medium">
                Learn More →
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;

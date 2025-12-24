# Parkinson's Disease Detection Platform

A comprehensive web application for early detection of Parkinson's disease through drawing and audio analysis using machine learning and modern web technologies.

## Features

- **User Authentication**: Secure login/signup with Firebase Authentication
- **Drawing Test**: Upload spiral drawings for motor function analysis
- **Audio Test**: Record or upload voice samples for vocal pattern analysis
- **AI Analysis**: Machine learning-powered prediction with confidence scores
- **Historical Tracking**: View confidence trends over time with interactive charts
- **Medical Information**: Comprehensive information about Parkinson's disease
- **Cloud Storage**: Secure file storage with Cloudinary integration
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS

## Tech Stack

- **Frontend**: React 19, Vite, Tailwind CSS
- **Authentication**: Firebase Auth
- **Database**: Firestore
- **File Storage**: Cloudinary
- **Charts**: Recharts
- **Icons**: Lucide React
- **Routing**: React Router DOM

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Firebase Configuration

1. Create a Firebase project at [https://console.firebase.google.com](https://console.firebase.google.com)
2. Enable Authentication (Email/Password)
3. Create a Firestore database
4. Update `src/firebase/config.js` with your Firebase configuration:

```javascript
const firebaseConfig = {
  apiKey: "your-api-key",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "your-app-id"
};
```

### 3. Cloudinary Configuration

1. Create a Cloudinary account at [https://cloudinary.com](https://cloudinary.com)
2. Update `src/cloudinary/config.js` with your Cloudinary credentials:

```javascript
cloudinary.config({
  cloud_name: 'your-cloud-name',
  api_key: 'your-api-key',
  api_secret: 'your-api-secret'
});
```

3. Set up upload presets in Cloudinary:
   - Create preset for images: `parkinsons-drawings`
   - Create preset for audio: `parkinsons-audio`

### 4. Firestore Security Rules

Add these security rules to your Firestore database:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /testResults/{document} {
      allow read, write: if request.auth != null && request.auth.uid == resource.data.userId;
    }
  }
}
```

### 5. Run the Application

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Project Structure

```
src/
├── components/          # Reusable components
│   ├── Navbar.jsx      # Navigation bar
│   └── ProtectedRoute.jsx # Route protection
├── contexts/           # React contexts
│   └── AuthContext.jsx # Authentication context
├── firebase/           # Firebase configuration
│   └── config.js       # Firebase setup
├── cloudinary/         # Cloudinary configuration
│   └── config.js       # Cloudinary setup
├── pages/              # Page components
│   ├── Login.jsx       # Login page
│   ├── Signup.jsx      # Signup page
│   ├── Home.jsx        # Home page
│   ├── DrawingTest.jsx # Drawing test page
│   ├── AudioTest.jsx   # Audio test page
│   ├── Results.jsx     # Results page
│   └── About.jsx       # About Parkinson's page
├── App.jsx             # Main app component
├── main.jsx           # Entry point
└── index.css          # Global styles
```

## Key Features Implementation

### Authentication
- Firebase Authentication with email/password
- Protected routes for authenticated users
- Automatic redirect to login for unauthenticated users

### File Upload & Storage
- Cloudinary integration for secure file storage
- Support for image and audio file uploads
- File validation and error handling

### Data Management
- Firestore for storing test results and user data
- Real-time data synchronization
- Historical data tracking

### UI/UX
- Responsive design with Tailwind CSS
- Medical-themed color scheme
- Accessible and user-friendly interface
- Interactive charts for data visualization

## Development Notes

- The application uses simulated AI analysis (random confidence scores)
- Replace the simulation with actual machine learning models
- Implement proper error handling and loading states
- Add comprehensive testing
- Consider implementing offline functionality

## Deployment

1. Build the application: `npm run build`
2. Deploy to your preferred hosting platform (Vercel, Netlify, etc.)
3. Update Firebase and Cloudinary configurations for production
4. Set up proper environment variables for sensitive data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions, please contact the development team or create an issue in the repository.
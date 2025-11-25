import { Beaker, Users, Code, Heart, Mail, Github, Twitter } from 'lucide-react';

const AboutPage = () => {
  return (
    <div className="container py-8">
      {/* Header */}
      <div className="mb-8 text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-accent-blue to-accent-teal rounded-full flex items-center justify-center mx-auto mb-4">
          <Beaker className="w-10 h-10 text-white" />
        </div>
        <h1 className="text-4xl font-bold text-white mb-3">
          About ToxPredict
        </h1>
        <p className="text-lg text-gray-400 max-w-2xl mx-auto">
          A research-grade platform for computational toxicity prediction using 
          Graph Convolutional Networks and machine learning
        </p>
      </div>

      {/* Mission Statement */}
      <div className="card mb-8 bg-gradient-to-br from-accent-blue/10 to-accent-teal/10 border-accent-blue/30">
        <h2 className="text-2xl font-bold text-white mb-4">
          Our Mission
        </h2>
        <p className="text-gray-300 leading-relaxed">
          ToxPredict aims to accelerate drug discovery and chemical safety assessment by providing 
          accurate, interpretable, and accessible toxicity predictions. We leverage state-of-the-art 
          deep learning techniques to analyze molecular structures and predict their potential 
          toxicological effects across multiple biological pathways.
        </p>
      </div>

      {/* Key Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="card">
          <div className="w-12 h-12 bg-accent-blue/20 rounded-lg flex items-center justify-center mb-3">
            <Code className="w-6 h-6 text-accent-blue" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">
            Advanced AI Models
          </h3>
          <p className="text-gray-400 text-sm">
            Our platform uses Graph Convolutional Networks (GCNs) that understand molecular 
            structure at the atomic level, plus a suite of baseline ML models (KNN, MLP, RF, SVM, XGBoost) 
            for comprehensive predictions.
          </p>
        </div>

        <div className="card">
          <div className="w-12 h-12 bg-accent-teal/20 rounded-lg flex items-center justify-center mb-3">
            <Beaker className="w-6 h-6 text-accent-teal" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">
            12 Toxicity Endpoints
          </h3>
          <p className="text-gray-400 text-sm">
            Predictions across multiple biological pathways including nuclear receptor activation 
            (NR-AhR, NR-AR, NR-ER, etc.) and stress response pathways (SR-ARE, SR-ATAD5, SR-HSE, etc.).
          </p>
        </div>

        <div className="card">
          <div className="w-12 h-12 bg-toxic-orange/20 rounded-lg flex items-center justify-center mb-3">
            <Users className="w-6 h-6 text-toxic-orange" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">
            Research-Grade Quality
          </h3>
          <p className="text-gray-400 text-sm">
            Built on the Tox21 dataset with rigorous validation, scaffold-based splitting to prevent 
            data leakage, and comprehensive performance metrics (AUROC, AUPRC, F1-Score).
          </p>
        </div>

        <div className="card">
          <div className="w-12 h-12 bg-safe-green/20 rounded-lg flex items-center justify-center mb-3">
            <Heart className="w-6 h-6 text-safe-green" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">
            Interpretable Results
          </h3>
          <p className="text-gray-400 text-sm">
            Beyond predictions, we provide molecular properties, confidence scores, and (coming soon) 
            attention visualizations to help you understand why the model makes specific predictions.
          </p>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-white mb-4">
          Technology Stack
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Frontend</p>
            <div className="space-y-1">
              <p className="text-sm text-gray-300">React 18</p>
              <p className="text-sm text-gray-300">TailwindCSS</p>
              <p className="text-sm text-gray-300">Vite</p>
            </div>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Backend</p>
            <div className="space-y-1">
              <p className="text-sm text-gray-300">FastAPI</p>
              <p className="text-sm text-gray-300">Python 3.8+</p>
              <p className="text-sm text-gray-300">Uvicorn</p>
            </div>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">ML/DL</p>
            <div className="space-y-1">
              <p className="text-sm text-gray-300">PyTorch</p>
              <p className="text-sm text-gray-300">scikit-learn</p>
              <p className="text-sm text-gray-300">XGBoost</p>
            </div>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Chemistry</p>
            <div className="space-y-1">
              <p className="text-sm text-gray-300">RDKit</p>
              <p className="text-sm text-gray-300">ECFP4</p>
              <p className="text-sm text-gray-300">SMILES</p>
            </div>
          </div>
        </div>
      </div>

      {/* Use Cases */}
      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-white mb-4">
          Use Cases
        </h2>
        <div className="space-y-3">
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-accent-blue/20 rounded flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-xs font-bold text-accent-blue">1</span>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-300">Drug Discovery</p>
              <p className="text-xs text-gray-500">
                Screen candidate molecules early in the drug development pipeline to identify potential toxicity issues
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-accent-teal/20 rounded flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-xs font-bold text-accent-teal">2</span>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-300">Chemical Safety Assessment</p>
              <p className="text-xs text-gray-500">
                Evaluate environmental chemicals and industrial compounds for regulatory compliance
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-toxic-orange/20 rounded flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-xs font-bold text-toxic-orange">3</span>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-300">Research & Education</p>
              <p className="text-xs text-gray-500">
                Study structure-activity relationships and teach computational toxicology concepts
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="card bg-toxic-red/10 border-toxic-red/30 mb-8">
        <h3 className="text-lg font-semibold text-toxic-red mb-2">
          ⚠️ Important Disclaimer
        </h3>
        <p className="text-sm text-gray-400">
          ToxPredict provides computational predictions for research and screening purposes only. 
          These predictions should NOT be used as sole evidence for regulatory submissions or clinical decisions. 
          Always validate predictions with experimental testing and consult with qualified toxicology experts. 
          The developers assume no liability for decisions made based on these predictions.
        </p>
      </div>

      {/* Contact & Links */}
      <div className="card bg-dark-bg">
        <h2 className="text-2xl font-bold text-white mb-4">
          Get in Touch
        </h2>
        <p className="text-gray-400 mb-4">
          Have questions, suggestions, or want to collaborate? We'd love to hear from you!
        </p>
        <div className="flex flex-wrap gap-3">
          <a href="mailto:contact@toxpredict.example.com" className="btn-secondary flex items-center space-x-2">
            <Mail className="w-4 h-4" />
            <span>Email Us</span>
          </a>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="btn-secondary flex items-center space-x-2">
            <Github className="w-4 h-4" />
            <span>GitHub</span>
          </a>
          <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="btn-secondary flex items-center space-x-2">
            <Twitter className="w-4 h-4" />
            <span>Twitter</span>
          </a>
        </div>
      </div>

      {/* Footer */}
      <div className="text-center mt-8 text-sm text-gray-500">
        <p>Built with ❤️ for the computational toxicology community</p>
        <p className="mt-1">© 2024 ToxPredict. All rights reserved.</p>
      </div>
    </div>
  );
};

export default AboutPage;

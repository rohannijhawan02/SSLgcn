import { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import { Search, Upload, CheckCircle, XCircle, Loader, Sparkles } from 'lucide-react';
import { toxicityAPI } from '../utils/api';
import { getErrorMessage } from '../utils/errorHandler';
import SMILESInput from '../components/SMILESInput';
import EndpointSelector from '../components/EndpointSelector';
import PredictionResults from '../components/PredictionResults';

const HomePage = () => {
  const [endpoints, setEndpoints] = useState([]);
  const [endpointPresets, setEndpointPresets] = useState({});
  const [selectedEndpoints, setSelectedEndpoints] = useState([]);
  const [smiles, setSMILES] = useState('');
  const [validationResult, setValidationResult] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResults, setPredictionResults] = useState(null);
  const [compareBaseline, setCompareBaseline] = useState(false);
  const [activeTab, setActiveTab] = useState('single'); // 'single', 'batch', 'draw'

  // Load endpoints on mount
  useEffect(() => {
    loadEndpoints();
  }, []);

  const loadEndpoints = async () => {
    try {
      const response = await toxicityAPI.getEndpoints();
      setEndpoints(response.data.endpoints);
      setEndpointPresets(response.data.presets);
    } catch (error) {
      const errorMessage = getErrorMessage(error, 'Failed to load toxicity endpoints');
      toast.error(errorMessage);
      console.error(error);
    }
  };

  const handleValidateSMILES = async () => {
    if (!smiles.trim()) {
      toast.error('Please enter a SMILES string');
      return;
    }

    setIsValidating(true);
    setValidationResult(null);

    try {
      const response = await toxicityAPI.validateSMILES(smiles.trim());
      setValidationResult(response.data);

      if (response.data.valid) {
        toast.success('Valid SMILES! Ready to predict.');
      } else {
        toast.error('Invalid SMILES string');
      }
    } catch (error) {
      const errorMessage = getErrorMessage(error, 'Validation failed');
      toast.error(errorMessage);
      setValidationResult({ valid: false, error: errorMessage });
    } finally {
      setIsValidating(false);
    }
  };

  const handlePredict = async () => {
    if (!validationResult || !validationResult.valid) {
      toast.error('Please validate SMILES first');
      return;
    }

    if (selectedEndpoints.length === 0) {
      toast.error('Please select at least one toxicity endpoint');
      return;
    }

    setIsPredicting(true);
    setPredictionResults(null);

    try {
      const response = await toxicityAPI.predictToxicity(
        smiles.trim(),
        selectedEndpoints,
        compareBaseline
      );
      setPredictionResults(response.data);
      toast.success('Prediction complete!');
    } catch (error) {
      const errorMessage = getErrorMessage(error, 'Prediction failed');
      toast.error(errorMessage);
      console.error(error);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleReset = () => {
    setSMILES('');
    setValidationResult(null);
    setPredictionResults(null);
    setSelectedEndpoints([]);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8 fade-in">
        <h1 className="text-4xl font-bold text-gray-100 mb-3">
          Toxicity Prediction
        </h1>
        <p className="text-lg text-gray-400">
          Predict molecular toxicity using Graph Convolutional Networks and baseline ML models
        </p>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Input Section */}
        <div className="lg:col-span-2 space-y-6">
          {/* Input Mode Tabs */}
          <div className="card">
            <div className="flex space-x-1 mb-6 border-b border-dark-border">
              <button
                className={`tab ${activeTab === 'single' ? 'tab-active' : ''}`}
                onClick={() => setActiveTab('single')}
              >
                <Search className="w-4 h-4 inline mr-2" />
                Single SMILES
              </button>
              <button
                className={`tab ${activeTab === 'batch' ? 'tab-active' : ''}`}
                onClick={() => setActiveTab('batch')}
                disabled
              >
                <Upload className="w-4 h-4 inline mr-2" />
                Batch Upload
                <span className="ml-2 text-xs bg-yellow-900/30 text-yellow-400 px-2 py-0.5 rounded">
                  Coming Soon
                </span>
              </button>
              <button
                className={`tab ${activeTab === 'draw' ? 'tab-active' : ''}`}
                onClick={() => setActiveTab('draw')}
                disabled
              >
                <Sparkles className="w-4 h-4 inline mr-2" />
                Draw Molecule
                <span className="ml-2 text-xs bg-yellow-900/30 text-yellow-400 px-2 py-0.5 rounded">
                  Coming Soon
                </span>
              </button>
            </div>

            {/* SMILES Input Component */}
            {activeTab === 'single' && (
              <SMILESInput
                smiles={smiles}
                setSMILES={setSMILES}
                validationResult={validationResult}
                isValidating={isValidating}
                onValidate={handleValidateSMILES}
                onReset={handleReset}
              />
            )}

            {/* Batch Upload (Placeholder) */}
            {activeTab === 'batch' && (
              <div className="text-center py-12 text-gray-400">
                <Upload className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium mb-2">Batch Upload</p>
                <p className="text-sm">Upload CSV or SMI files with multiple compounds</p>
                <p className="text-xs mt-4 text-gray-500">Feature coming soon...</p>
              </div>
            )}

            {/* Molecule Drawing (Placeholder) */}
            {activeTab === 'draw' && (
              <div className="text-center py-12 text-gray-400">
                <Sparkles className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium mb-2">Draw Molecule</p>
                <p className="text-sm">Interactive molecule drawing tool</p>
                <p className="text-xs mt-4 text-gray-500">Feature coming soon...</p>
              </div>
            )}
          </div>

          {/* Molecule Visualization */}
          {validationResult && validationResult.valid && (
            <div className="card fade-in">
              <h3 className="text-lg font-semibold text-gray-100 mb-4 flex items-center">
                <CheckCircle className="w-5 h-5 text-safe-green mr-2" />
                Validated Molecule
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Molecule Image */}
                {validationResult.image && (
                  <div className="flex justify-center items-center bg-white rounded-lg p-4">
                    <img
                      src={validationResult.image}
                      alt="Molecule Structure"
                      className="max-w-full h-auto"
                    />
                  </div>
                )}

                {/* Molecular Properties */}
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">
                      Canonical SMILES
                    </p>
                    <p className="text-sm font-mono bg-dark-bg px-3 py-2 rounded border border-dark-border break-all">
                      {validationResult.canonical_smiles}
                    </p>
                  </div>

                  <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">
                      Molecular Formula
                    </p>
                    <p className="text-sm font-mono bg-dark-bg px-3 py-2 rounded border border-dark-border">
                      {validationResult.molecular_formula}
                    </p>
                  </div>

                  {validationResult.properties && (
                    <div className="grid grid-cols-2 gap-2 pt-2">
                      {Object.entries(validationResult.properties).slice(0, 4).map(([key, value]) => (
                        <div key={key} className="bg-dark-bg px-3 py-2 rounded border border-dark-border">
                          <p className="text-xs text-gray-400 capitalize">
                            {key.replace(/_/g, ' ')}
                          </p>
                          <p className="text-sm font-semibold text-gray-100">
                            {typeof value === 'number' ? value.toFixed(2) : value}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Prediction Results */}
          {predictionResults && (
            <PredictionResults results={predictionResults} />
          )}
        </div>

        {/* Right Column - Endpoint Selection & Prediction */}
        <div className="space-y-6">
          {/* Endpoint Selector */}
          <EndpointSelector
            endpoints={endpoints}
            selectedEndpoints={selectedEndpoints}
            setSelectedEndpoints={setSelectedEndpoints}
            presets={endpointPresets}
          />

          {/* Options */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-100 mb-4">Options</h3>
            <label className="flex items-center space-x-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={compareBaseline}
                onChange={(e) => setCompareBaseline(e.target.checked)}
                className="w-5 h-5 rounded border-dark-border bg-dark-bg text-accent-blue 
                         focus:ring-2 focus:ring-accent-blue focus:ring-offset-0 
                         focus:ring-offset-dark-card transition-all cursor-pointer"
              />
              <span className="text-sm text-gray-300 group-hover:text-gray-100 transition-colors">
                Compare with Baseline Models
                <span className="block text-xs text-gray-500 mt-0.5">
                  (KNN, NN, RF, SVM, XGBoost)
                </span>
              </span>
            </label>
          </div>

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={!validationResult?.valid || selectedEndpoints.length === 0 || isPredicting}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            {isPredicting ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                <span>Predicting...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                <span>Run Prediction</span>
              </>
            )}
          </button>

          {/* Info Card */}
          <div className="card bg-blue-900/10 border-blue-800">
            <h4 className="text-sm font-semibold text-blue-400 mb-2">ℹ️ How it works</h4>
            <ul className="text-xs text-gray-400 space-y-2">
              <li>1. Enter or paste a SMILES string</li>
              <li>2. Validate the molecular structure</li>
              <li>3. Select toxicity endpoints</li>
              <li>4. Run prediction with GCN model</li>
              <li>5. Optionally compare with baseline ML models</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;

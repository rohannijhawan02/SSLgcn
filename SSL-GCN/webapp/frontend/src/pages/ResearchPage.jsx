import { useState, useEffect } from 'react';
import { 
  FileText, BookOpen, TrendingUp, Award, Download, BarChart3, 
  LineChart, Activity, CheckCircle, AlertCircle 
} from 'lucide-react';
import { toxicityAPI } from '../utils/api';
import ROCCurveChart from '../components/ROCCurveChart';
import PerformanceTable from '../components/PerformanceTable';

const ResearchPage = () => {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [selectedToxicity, setSelectedToxicity] = useState('NR-AhR');
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      const response = await toxicityAPI.getResearchMetrics();
      setMetrics(response.data);
    } catch (error) {
      console.error('Error loading metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="container py-8">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent mx-auto mb-4"></div>
            <p className="text-gray-400">Loading research metrics...</p>
          </div>
        </div>
      </div>
    );
  }

  const gcnResults = metrics?.gcn_results || [];
  const baselineResults = metrics?.baseline_results || {};
  const rocData = metrics?.roc_data || {};
  const availableToxicities = metrics?.available_toxicities || [];

  // Calculate average metrics
  const avgGCNAUC = gcnResults.length > 0 
    ? (gcnResults.reduce((sum, r) => sum + r.test_roc_auc, 0) / gcnResults.length).toFixed(3)
    : '0.000';
  
  const avgGCNF1 = gcnResults.length > 0 
    ? (gcnResults.reduce((sum, r) => sum + r.test_f1, 0) / gcnResults.length).toFixed(3)
    : '0.000';

  return (
    <div className="container py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-3">
          Research & Performance Metrics
        </h1>
        <p className="text-lg text-gray-400">
          Comprehensive evaluation of GCN and baseline models across 12 toxicity endpoints
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="glass-panel p-4">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-8 h-8 text-accent-teal" />
            <span className="text-2xl font-bold text-white">{gcnResults.length}</span>
          </div>
          <p className="text-sm text-gray-400">Toxicity Endpoints</p>
          <p className="text-xs text-gray-500 mt-1">Nuclear Receptor & Stress Response</p>
        </div>

        <div className="glass-panel p-4">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-8 h-8 text-safe-green" />
            <span className="text-2xl font-bold text-white">{avgGCNAUC}</span>
          </div>
          <p className="text-sm text-gray-400">Avg GCN ROC-AUC</p>
          <p className="text-xs text-gray-500 mt-1">Test set performance</p>
        </div>

        <div className="glass-panel p-4">
          <div className="flex items-center justify-between mb-2">
            <CheckCircle className="w-8 h-8 text-accent-blue" />
            <span className="text-2xl font-bold text-white">{avgGCNF1}</span>
          </div>
          <p className="text-sm text-gray-400">Avg GCN F1-Score</p>
          <p className="text-xs text-gray-500 mt-1">Balanced performance</p>
        </div>

        <div className="glass-panel p-4">
          <div className="flex items-center justify-between mb-2">
            <BarChart3 className="w-8 h-8 text-toxic-orange" />
            <span className="text-2xl font-bold text-white">{availableToxicities.length}</span>
          </div>
          <p className="text-sm text-gray-400">Baseline Models</p>
          <p className="text-xs text-gray-500 mt-1">Trained toxicities</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="glass-panel mb-6">
        <div className="flex space-x-1 border-b border-dark-border overflow-x-auto">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-6 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
              activeTab === 'overview'
                ? 'text-accent border-b-2 border-accent'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            Overview
          </button>
          <button
            onClick={() => setActiveTab('gcn')}
            className={`px-6 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
              activeTab === 'gcn'
                ? 'text-accent border-b-2 border-accent'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            <TrendingUp className="w-4 h-4 inline mr-2" />
            GCN Results
          </button>
          <button
            onClick={() => setActiveTab('baseline')}
            className={`px-6 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
              activeTab === 'baseline'
                ? 'text-accent border-b-2 border-accent'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            <LineChart className="w-4 h-4 inline mr-2" />
            Baseline Models
          </button>
          <button
            onClick={() => setActiveTab('roc')}
            className={`px-6 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
              activeTab === 'roc'
                ? 'text-accent border-b-2 border-accent'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            <Activity className="w-4 h-4 inline mr-2" />
            ROC Curves
          </button>
          <button
            onClick={() => setActiveTab('methodology')}
            className={`px-6 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
              activeTab === 'methodology'
                ? 'text-accent border-b-2 border-accent'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            <BookOpen className="w-4 h-4 inline mr-2" />
            Methodology
          </button>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          <PerformanceTable 
            gcnResults={gcnResults}
            baselineResults={baselineResults}
          />
        </div>
      )}

      {activeTab === 'gcn' && (
        <div className="space-y-6">
          <div className="glass-panel p-6">
            <h3 className="text-xl font-semibold text-white mb-4">
              GCN Model Performance Across All Endpoints
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-border">
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Toxicity</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Train</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Test</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">ROC-AUC</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Accuracy</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Precision</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Recall</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">F1-Score</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Best Epoch</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-border">
                  {gcnResults.map((result, idx) => (
                    <tr key={idx} className="hover:bg-dark-bg transition-colors">
                      <td className="px-4 py-3 text-sm font-medium text-gray-300">{result.toxicity}</td>
                      <td className="px-4 py-3 text-center text-sm text-gray-400">{result.train_samples}</td>
                      <td className="px-4 py-3 text-center text-sm text-gray-400">{result.test_samples}</td>
                      <td className="px-4 py-3 text-center">
                        <span className={`font-mono text-sm ${
                          result.test_roc_auc >= 0.8 ? 'text-safe-green' :
                          result.test_roc_auc >= 0.7 ? 'text-yellow-500' :
                          'text-toxic-red'
                        }`}>
                          {result.test_roc_auc.toFixed(3)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                        {result.test_accuracy.toFixed(3)}
                      </td>
                      <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                        {result.test_precision.toFixed(3)}
                      </td>
                      <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                        {result.test_recall.toFixed(3)}
                      </td>
                      <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                        {result.test_f1.toFixed(3)}
                      </td>
                      <td className="px-4 py-3 text-center text-sm text-gray-400">{result.best_epoch}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* GCN Architecture Info */}
          <div className="glass-panel p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Model Architecture</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <p className="text-xs text-gray-400 mb-1">Model Type</p>
                <p className="text-sm font-medium text-white">Graph Convolutional Network (GCN)</p>
              </div>
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <p className="text-xs text-gray-400 mb-1">Input Features</p>
                <p className="text-sm font-medium text-white">74-dimensional atom features</p>
              </div>
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <p className="text-xs text-gray-400 mb-1">Architecture</p>
                <p className="text-sm font-medium text-white">3-layer GCN [64, 128, 256]</p>
              </div>
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <p className="text-xs text-gray-400 mb-1">Output</p>
                <p className="text-sm font-medium text-white">Binary Classification (Toxic/Non-toxic)</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'baseline' && (
        <div className="space-y-6">
          {availableToxicities.length > 0 ? (
            availableToxicities.map((toxicity) => (
              <div key={toxicity} className="glass-panel p-6">
                <h3 className="text-xl font-semibold text-white mb-4">{toxicity}</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-dark-border">
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Model</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">CV ROC-AUC</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Test ROC-AUC</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Accuracy</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Precision</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">Recall</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase">F1-Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-dark-border">
                      {baselineResults[toxicity]?.map((model, idx) => (
                        <tr key={idx} className="hover:bg-dark-bg transition-colors">
                          <td className="px-4 py-3 text-sm font-medium text-gray-300">{model.model}</td>
                          <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                            {model.cv_roc_auc.toFixed(3)}
                          </td>
                          <td className="px-4 py-3 text-center">
                            <span className={`font-mono text-sm ${
                              model.test_roc_auc >= 0.8 ? 'text-safe-green' :
                              model.test_roc_auc >= 0.7 ? 'text-yellow-500' :
                              'text-toxic-red'
                            }`}>
                              {model.test_roc_auc.toFixed(3)}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                            {model.test_accuracy.toFixed(3)}
                          </td>
                          <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                            {model.test_precision.toFixed(3)}
                          </td>
                          <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                            {model.test_recall.toFixed(3)}
                          </td>
                          <td className="px-4 py-3 text-center text-sm font-mono text-gray-300">
                            {model.test_f1.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))
          ) : (
            <div className="glass-panel p-8 text-center">
              <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
              <p className="text-gray-400">No baseline model results available</p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'roc' && (
        <div className="space-y-6">
          {Object.keys(rocData).length > 0 ? (
            <>
              {/* Toxicity Selector */}
              <div className="glass-panel p-4">
                <label className="text-sm font-medium text-gray-400 mb-2 block">
                  Select Toxicity Endpoint:
                </label>
                <select
                  value={selectedToxicity}
                  onChange={(e) => setSelectedToxicity(e.target.value)}
                  className="w-full md:w-64 px-4 py-2 bg-dark-bg border border-dark-border rounded-lg text-white focus:outline-none focus:border-accent"
                >
                  {Object.keys(rocData).map((tox) => (
                    <option key={tox} value={tox}>{tox}</option>
                  ))}
                </select>
              </div>

              {/* ROC Curve */}
              <ROCCurveChart 
                toxicity={selectedToxicity}
                rocData={rocData[selectedToxicity]}
              />

              {/* ROC Interpretation */}
              <div className="glass-panel p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Understanding ROC Curves</h3>
                <div className="space-y-3 text-sm text-gray-400">
                  <p>
                    <strong className="text-white">ROC (Receiver Operating Characteristic) Curve:</strong> Shows
                    the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)
                    at various classification thresholds.
                  </p>
                  <p>
                    <strong className="text-white">AUC (Area Under Curve):</strong> Ranges from 0 to 1, where:
                  </p>
                  <ul className="list-disc list-inside pl-4 space-y-1">
                    <li>AUC = 1.0: Perfect classifier</li>
                    <li>AUC ≥ 0.8: Excellent performance</li>
                    <li>AUC ≥ 0.7: Good performance</li>
                    <li>AUC = 0.5: No better than random guessing</li>
                  </ul>
                  <p>
                    <strong className="text-white">Interpretation:</strong> The closer the curve is to the top-left
                    corner, the better the model's performance. The diagonal line represents random guessing.
                  </p>
                </div>
              </div>
            </>
          ) : (
            <div className="glass-panel p-8 text-center">
              <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
              <p className="text-gray-400">No ROC curve data available</p>
              <p className="text-sm text-gray-500 mt-2">Train baseline models to generate ROC curves</p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'methodology' && (
        <div className="space-y-6">
          {/* Dataset */}
          <div className="glass-panel p-6">
            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 bg-toxic-orange/20 rounded-lg flex items-center justify-center flex-shrink-0">
                <Award className="w-6 h-6 text-toxic-orange" />
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-white mb-2">
                  Dataset & Validation
                </h3>
                <p className="text-gray-400 mb-3">
                  The Tox21 Challenge dataset, focusing on nuclear receptor and stress response pathways.
                </p>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <strong>Source:</strong> Tox21 Data Challenge 2014</li>
                  <li>• <strong>Compounds:</strong> ~8,000 unique chemical structures</li>
                  <li>• <strong>Endpoints:</strong> 12 toxicity pathways (7 Nuclear Receptor, 5 Stress Response)</li>
                  <li>• <strong>Split Strategy:</strong> Scaffold-based splitting to prevent data leakage</li>
                  <li>• <strong>Ratio:</strong> 80% train, 10% validation, 10% test</li>
                </ul>
              </div>
            </div>
          </div>

          {/* GCN Model */}
          <div className="glass-panel p-6">
            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 bg-accent-blue/20 rounded-lg flex items-center justify-center flex-shrink-0">
                <BookOpen className="w-6 h-6 text-accent-blue" />
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-white mb-2">
                  GCN Architecture
                </h3>
                <p className="text-gray-400 mb-3">
                  Graph Convolutional Network for learning molecular representations directly from chemical structures.
                </p>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <strong>Input:</strong> Molecular graphs with 74-dimensional atom features</li>
                  <li>• <strong>Architecture:</strong> 3-layer GCN with hidden dimensions [64, 128, 256]</li>
                  <li>• <strong>Aggregation:</strong> Mean pooling for graph-level representations</li>
                  <li>• <strong>Classifier:</strong> 2-layer MLP with 128 hidden units</li>
                  <li>• <strong>Training:</strong> Adam optimizer, cross-entropy loss, dropout 0.3</li>
                  <li>• <strong>Early Stopping:</strong> Patience of 50 epochs on validation AUC</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Baseline Models */}
          <div className="glass-panel p-6">
            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 bg-safe-green/20 rounded-lg flex items-center justify-center flex-shrink-0">
                <TrendingUp className="w-6 h-6 text-safe-green" />
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-white mb-2">
                  Baseline ML Models
                </h3>
                <p className="text-gray-400 mb-3">
                  Traditional machine learning models using ECFP4 molecular fingerprints for comparison.
                </p>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• <strong>Features:</strong> ECFP4 (Extended Connectivity Fingerprints, 2048 bits, radius 2)</li>
                  <li>• <strong>Models:</strong> KNN, Neural Network, Random Forest, SVM, XGBoost</li>
                  <li>• <strong>Hyperparameter Tuning:</strong> 5-fold cross-validation with Bayesian optimization</li>
                  <li>• <strong>Evaluation:</strong> Multiple metrics (ROC-AUC, AUPRC, F1, Accuracy, Precision, Recall)</li>
                  <li>• <strong>Cross-Validation:</strong> Stratified K-Fold to handle class imbalance</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="glass-panel p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Performance Metrics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <p className="text-sm font-medium text-white mb-2">ROC-AUC (Area Under ROC Curve)</p>
                <p className="text-xs text-gray-400">
                  Measures the model's ability to distinguish between toxic and non-toxic compounds
                  across all classification thresholds. Primary metric for imbalanced datasets.
                </p>
              </div>
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <p className="text-sm font-medium text-white mb-2">F1-Score</p>
                <p className="text-xs text-gray-400">
                  Harmonic mean of precision and recall, providing a balanced measure of model
                  performance for imbalanced classes.
                </p>
              </div>
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <p className="text-sm font-medium text-white mb-2">Precision</p>
                <p className="text-xs text-gray-400">
                  Proportion of predicted toxic compounds that are actually toxic.
                  Important for minimizing false alarms.
                </p>
              </div>
              <div className="p-4 bg-dark-bg rounded-lg border border-dark-border">
                <p className="text-sm font-medium text-white mb-2">Recall (Sensitivity)</p>
                <p className="text-xs text-gray-400">
                  Proportion of actual toxic compounds correctly identified.
                  Critical for safety screening applications.
                </p>
              </div>
            </div>
          </div>

          {/* Key Publications */}
          <div className="glass-panel p-6">
            <h3 className="text-xl font-semibold text-white mb-4">
              📖 Key Publications
            </h3>
            <div className="space-y-4">
              <div className="border-l-2 border-accent-blue pl-4">
                <p className="text-sm font-medium text-gray-300">
                  Tox21 Challenge: Building Predictive Models of Nuclear Receptor and Stress Response Pathways
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Huang R. et al., Front. Environ. Sci., 2016
                </p>
              </div>
              <div className="border-l-2 border-accent-teal pl-4">
                <p className="text-sm font-medium text-gray-300">
                  Semi-Supervised Classification with Graph Convolutional Networks
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Kipf & Welling, ICLR, 2017
                </p>
              </div>
              <div className="border-l-2 border-toxic-orange pl-4">
                <p className="text-sm font-medium text-gray-300">
                  Analyzing Learned Molecular Representations for Property Prediction
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Yang et al., J. Chem. Inf. Model., 2019
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Download Section */}
      <div className="glass-panel p-6 bg-dark-bg mt-8">
        <h3 className="text-lg font-semibold text-white mb-4">
          📥 Download Resources
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <button
            onClick={() => {
              const dataStr = JSON.stringify(metrics, null, 2);
              const dataBlob = new Blob([dataStr], { type: 'application/json' });
              const url = URL.createObjectURL(dataBlob);
              const link = document.createElement('a');
              link.href = url;
              link.download = 'research_metrics.json';
              link.click();
            }}
            className="btn-secondary"
          >
            <Download className="w-4 h-4 mr-2" />
            Metrics (JSON)
          </button>
          <button
            onClick={() => {
              // Convert GCN results to CSV
              let csv = 'Toxicity,Train,Test,ROC-AUC,Accuracy,Precision,Recall,F1,Best Epoch\n';
              gcnResults.forEach(r => {
                csv += `${r.toxicity},${r.train_samples},${r.test_samples},${r.test_roc_auc},${r.test_accuracy},${r.test_precision},${r.test_recall},${r.test_f1},${r.best_epoch}\n`;
              });
              const blob = new Blob([csv], { type: 'text/csv' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = 'gcn_results.csv';
              link.click();
            }}
            className="btn-secondary"
          >
            <FileText className="w-4 h-4 mr-2" />
            GCN Results (CSV)
          </button>
          <button
            onClick={() => {
              const blob = new Blob([JSON.stringify(rocData, null, 2)], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = 'roc_data.json';
              link.click();
            }}
            className="btn-secondary"
          >
            <TrendingUp className="w-4 h-4 mr-2" />
            ROC Data (JSON)
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResearchPage;

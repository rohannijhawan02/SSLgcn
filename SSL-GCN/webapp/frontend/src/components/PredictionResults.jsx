import { Download, AlertCircle, CheckCircle, XCircle, BarChart3 } from 'lucide-react';
import { useState } from 'react';

const PredictionResults = ({ results }) => {
  const [showComparison, setShowComparison] = useState(true);

  if (!results || !results.predictions) {
    return null;
  }

  const { predictions, baseline_predictions, compare_baseline, properties, smiles, canonical_smiles } = results;

  // Calculate overall risk from GCN predictions
  const toxicCount = predictions.filter((p) => p.prediction === 'Toxic').length;
  const totalCount = predictions.length;
  const toxicPercentage = ((toxicCount / totalCount) * 100).toFixed(1);

  const handleDownloadJSON = () => {
    const data = {
      smiles,
      canonical_smiles,
      properties,
      gcn_predictions: predictions,
      baseline_predictions: baseline_predictions || null,
      timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `toxicity_prediction_${new Date().getTime()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <div className="glass-panel p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gradient mb-2">Prediction Results</h2>
            <p className="text-sm text-gray-400">
              {predictions[0]?.is_mock ? 'Demo Mode (Install PyTorch for real predictions)' : 'Using Trained GCN Models'}
            </p>
          </div>
          <button
            onClick={handleDownloadJSON}
            className="btn-secondary flex items-center space-x-2"
          >
            <Download className="w-4 h-4" />
            <span>Export JSON</span>
          </button>
        </div>

        {/* Overall Risk Assessment */}
        <div className={`p-4 rounded-lg border-2 ${
          toxicPercentage > 50
            ? 'bg-toxic-red/10 border-toxic-red/30'
            : 'bg-safe-green/10 border-safe-green/30'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {toxicPercentage > 50 ? (
                <AlertCircle className="w-8 h-8 text-toxic-red" />
              ) : (
                <CheckCircle className="w-8 h-8 text-safe-green" />
              )}
              <div>
                <h3 className="text-lg font-semibold text-white">
                  Overall Risk: {toxicPercentage > 50 ? 'HIGH' : 'LOW'}
                </h3>
                <p className="text-sm text-gray-400">
                  {toxicCount} of {totalCount} endpoints show toxicity ({toxicPercentage}%)
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-white">{toxicPercentage}%</div>
              <div className="text-xs text-gray-400">Toxic Probability</div>
            </div>
          </div>
        </div>
      </div>

      {/* Molecular Properties */}
      {properties && (
        <div className="glass-panel p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2" />
            Molecular Properties
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 bg-dark-bg rounded-lg border border-dark-border">
              <div className="text-xs text-gray-400 mb-1">Molecular Weight</div>
              <div className="text-lg font-semibold text-white">{properties.molecular_weight?.toFixed(2)}</div>
            </div>
            <div className="p-3 bg-dark-bg rounded-lg border border-dark-border">
              <div className="text-xs text-gray-400 mb-1">LogP</div>
              <div className="text-lg font-semibold text-white">{properties.logP?.toFixed(2)}</div>
            </div>
            <div className="p-3 bg-dark-bg rounded-lg border border-dark-border">
              <div className="text-xs text-gray-400 mb-1">H-Bond Donors</div>
              <div className="text-lg font-semibold text-white">{properties.num_h_donors}</div>
            </div>
            <div className="p-3 bg-dark-bg rounded-lg border border-dark-border">
              <div className="text-xs text-gray-400 mb-1">H-Bond Acceptors</div>
              <div className="text-lg font-semibold text-white">{properties.num_h_acceptors}</div>
            </div>
            <div className="p-3 bg-dark-bg rounded-lg border border-dark-border">
              <div className="text-xs text-gray-400 mb-1">Rotatable Bonds</div>
              <div className="text-lg font-semibold text-white">{properties.num_rotatable_bonds}</div>
            </div>
            <div className="p-3 bg-dark-bg rounded-lg border border-dark-border">
              <div className="text-xs text-gray-400 mb-1">Aromatic Rings</div>
              <div className="text-lg font-semibold text-white">{properties.num_aromatic_rings}</div>
            </div>
            <div className="p-3 bg-dark-bg rounded-lg border border-dark-border">
              <div className="text-xs text-gray-400 mb-1">TPSA</div>
              <div className="text-lg font-semibold text-white">{properties.tpsa?.toFixed(2)}</div>
            </div>
            <div className="p-3 bg-dark-bg rounded-lg border border-dark-border">
              <div className="text-xs text-gray-400 mb-1">Formal Charge</div>
              <div className="text-lg font-semibold text-white">{properties.formal_charge}</div>
            </div>
          </div>
        </div>
      )}

      {/* GCN Predictions Table */}
      <div className="glass-panel p-6">
        <h3 className="text-lg font-semibold text-white mb-4">GCN Model Predictions</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-dark-border">
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Endpoint
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Prediction
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Probability
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Confidence
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-dark-border">
              {predictions.map((pred, index) => (
                <tr key={index} className="hover:bg-dark-bg transition-colors">
                  <td className="px-4 py-3 text-sm font-medium text-gray-300">
                    {pred.endpoint_name || pred.endpoint}
                  </td>

                  <td className="px-4 py-3 text-xs text-gray-400">
                    {pred.category}
                  </td>

                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${
                      pred.prediction === 'Toxic'
                        ? 'bg-toxic-red/20 text-toxic-red border border-toxic-red/30'
                        : 'bg-safe-green/20 text-safe-green border border-safe-green/30'
                    }`}>
                      {pred.prediction === 'Toxic' ? (
                        <XCircle className="w-3 h-3 mr-1" />
                      ) : (
                        <CheckCircle className="w-3 h-3 mr-1" />
                      )}
                      {pred.prediction}
                    </span>
                  </td>

                  <td className="px-4 py-3 text-sm font-mono text-gray-300">
                    {(pred.probability * 100).toFixed(2)}%
                  </td>

                  <td className="px-4 py-3">
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 h-2 bg-dark-border rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            pred.confidence >= 0.8 ? 'bg-safe-green' : pred.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-toxic-red'
                          }`}
                          style={{ width: `${pred.confidence * 100}%` }}
                        />
                      </div>
                      <span className={`text-xs font-medium ${
                        pred.confidence >= 0.8 ? 'text-safe-green' : pred.confidence >= 0.6 ? 'text-yellow-500' : 'text-toxic-red'
                      }`}>
                        {pred.confidence_level}
                      </span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Baseline Model Comparison */}
      {compare_baseline && (
        <div className="glass-panel p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white">Model Comparison</h3>
              {baseline_predictions && (
                <p className="text-xs text-gray-400 mt-1">
                  Trained models available for: {baseline_predictions.filter(bp => bp.is_trained).map(bp => bp.endpoint).join(', ') || 'None'}
                </p>
              )}
            </div>
            <button
              onClick={() => setShowComparison(!showComparison)}
              className="text-sm text-accent hover:text-accent/80 transition-colors"
            >
              {showComparison ? 'Hide' : 'Show'} Table
            </button>
          </div>

          {showComparison && (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-border">
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Endpoint
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                      GCN
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                      XGBoost
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Random Forest
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                      SVM
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Neural Network
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase tracking-wider">
                      KNN
                    </th>
                  </tr>
                </thead>

                <tbody className="divide-y divide-dark-border">
                  {predictions.map((pred, index) => {
                    // Find corresponding baseline predictions for this endpoint
                    const baselinePred = baseline_predictions ? 
                      baseline_predictions.find(bp => bp.endpoint === pred.endpoint) : null;
                    
                    const isTrained = baselinePred && baselinePred.is_trained;
                    const models = baselinePred?.models || {};
                    
                    const renderCell = (modelType) => {
                      if (!isTrained || !models[modelType]) {
                        return <td key={modelType} className="px-4 py-3 text-center text-gray-500">-</td>;
                      }
                      
                      const modelPred = models[modelType];
                      return (
                        <td key={modelType} className="px-4 py-3 text-center">
                          <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                            modelPred.prediction === 'Toxic'
                              ? 'bg-toxic-red/20 text-toxic-red'
                              : 'bg-safe-green/20 text-safe-green'
                          }`}>
                            {modelPred.prediction}
                          </span>
                        </td>
                      );
                    };
                    
                    return (
                      <tr key={index} className="hover:bg-dark-bg transition-colors">
                        <td className="px-4 py-3 text-sm font-medium text-gray-300">
                          {pred.endpoint_name || pred.endpoint}
                          {isTrained && (
                            <span className="ml-2 text-xs text-safe-green">●</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                            pred.prediction === 'Toxic'
                              ? 'bg-toxic-red/20 text-toxic-red'
                              : 'bg-safe-green/20 text-safe-green'
                          }`}>
                            {pred.prediction}
                          </span>
                        </td>
                        {renderCell('XGBoost')}
                        {renderCell('RF')}
                        {renderCell('SVM')}
                        {renderCell('NN')}
                        {renderCell('KNN')}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          
          {showComparison && (
            <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <p className="text-xs text-gray-300">
                <strong className="text-blue-400">Legend:</strong> <span className="text-safe-green">●</span> = Trained models available for this toxicity. 
                "-" indicates no trained model for that toxicity/model combination.
              </p>
            </div>
          )}
        </div>
      )}

      {/* SMILES Information */}
      <div className="glass-panel p-6">
        <h3 className="text-lg font-semibold text-white mb-3">Molecular Structure</h3>
        <div className="space-y-2">
          <div>
            <span className="text-xs text-gray-400">Input SMILES:</span>
            <p className="text-sm font-mono text-gray-300 bg-dark-bg p-2 rounded mt-1 break-all">
              {smiles}
            </p>
          </div>
          {canonical_smiles && canonical_smiles !== smiles && (
            <div>
              <span className="text-xs text-gray-400">Canonical SMILES:</span>
              <p className="text-sm font-mono text-gray-300 bg-dark-bg p-2 rounded mt-1 break-all">
                {canonical_smiles}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;

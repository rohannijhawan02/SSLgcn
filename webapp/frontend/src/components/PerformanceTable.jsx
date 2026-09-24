const PerformanceTable = ({ gcnResults, baselineResults }) => {
  // Get all toxicities
  const allToxicities = gcnResults.map(r => r.toxicity);
  
  // Get baseline toxicities
  const baselineToxicities = Object.keys(baselineResults);

  return (
    <div className="glass-panel p-6">
      <h3 className="text-xl font-semibold text-white mb-4">
        Comprehensive Model Performance Comparison
      </h3>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-dark-border">
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase sticky left-0 bg-dark-card z-10">
                Toxicity
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase" colSpan="2">
                GCN
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase" colSpan="2">
                XGBoost
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase" colSpan="2">
                Random Forest
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase" colSpan="2">
                SVM
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase" colSpan="2">
                Neural Network
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-gray-400 uppercase" colSpan="2">
                KNN
              </th>
            </tr>
            <tr className="border-b border-dark-border">
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 sticky left-0 bg-dark-card z-10"></th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">AUC</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">F1</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">AUC</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">F1</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">AUC</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">F1</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">AUC</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">F1</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">AUC</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">F1</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">AUC</th>
              <th className="px-2 py-2 text-center text-xs font-medium text-gray-500">F1</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-dark-border">
            {allToxicities.map((toxicity) => {
              const gcn = gcnResults.find(r => r.toxicity === toxicity);
              const baseline = baselineResults[toxicity];
              
              const getModelData = (modelName) => {
                if (!baseline) return null;
                return baseline.find(m => m.model === modelName);
              };
              
              const xgboost = getModelData('XGBoost');
              const rf = getModelData('RF');
              const svm = getModelData('SVM');
              const nn = getModelData('NN');
              const knn = getModelData('KNN');
              
              const renderMetric = (value, isHighlighted = false) => {
                if (value === null || value === undefined) {
                  return <span className="text-gray-600">-</span>;
                }
                
                const colorClass = value >= 0.8 ? 'text-safe-green' : 
                                  value >= 0.7 ? 'text-yellow-500' : 
                                  'text-gray-300';
                
                return (
                  <span className={`font-mono text-sm ${isHighlighted ? 'font-bold' : ''} ${colorClass}`}>
                    {value.toFixed(3)}
                  </span>
                );
              };
              
              return (
                <tr key={toxicity} className="hover:bg-dark-bg transition-colors">
                  <td className="px-4 py-3 text-sm font-medium text-gray-300 sticky left-0 bg-dark-card z-10">
                    {toxicity}
                    {baseline && (
                      <span className="ml-2 text-xs text-safe-green">●</span>
                    )}
                  </td>
                  {/* GCN */}
                  <td className="px-2 py-3 text-center">
                    {renderMetric(gcn?.test_roc_auc)}
                  </td>
                  <td className="px-2 py-3 text-center">
                    {renderMetric(gcn?.test_f1)}
                  </td>
                  {/* XGBoost */}
                  <td className="px-2 py-3 text-center">
                    {renderMetric(xgboost?.test_roc_auc)}
                  </td>
                  <td className="px-2 py-3 text-center">
                    {renderMetric(xgboost?.test_f1)}
                  </td>
                  {/* RF */}
                  <td className="px-2 py-3 text-center">
                    {renderMetric(rf?.test_roc_auc)}
                  </td>
                  <td className="px-2 py-3 text-center">
                    {renderMetric(rf?.test_f1)}
                  </td>
                  {/* SVM */}
                  <td className="px-2 py-3 text-center">
                    {renderMetric(svm?.test_roc_auc)}
                  </td>
                  <td className="px-2 py-3 text-center">
                    {renderMetric(svm?.test_f1)}
                  </td>
                  {/* NN */}
                  <td className="px-2 py-3 text-center">
                    {renderMetric(nn?.test_roc_auc)}
                  </td>
                  <td className="px-2 py-3 text-center">
                    {renderMetric(nn?.test_f1)}
                  </td>
                  {/* KNN */}
                  <td className="px-2 py-3 text-center">
                    {renderMetric(knn?.test_roc_auc)}
                  </td>
                  <td className="px-2 py-3 text-center">
                    {renderMetric(knn?.test_f1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      
      <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
        <p className="text-xs text-gray-300">
          <strong className="text-blue-400">Legend:</strong>{' '}
          <span className="text-safe-green">Green</span> = AUC/F1 ≥ 0.8 (Excellent),{' '}
          <span className="text-yellow-500">Yellow</span> = 0.7-0.8 (Good),{' '}
          <span className="text-safe-green">●</span> = Baseline models trained for this toxicity,{' '}
          "-" = No trained model available
        </p>
      </div>
    </div>
  );
};

export default PerformanceTable;

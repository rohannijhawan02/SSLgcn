import { Brain, TrendingUp, Target, Sparkles } from 'lucide-react';

const ExplainabilityPage = () => {
  return (
    <div className="container py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-3">
          Model Explainability
        </h1>
        <p className="text-lg text-gray-400">
          Understanding the AI: Interpretability, attention visualization, and feature importance
        </p>
      </div>

      {/* Coming Soon Banner */}
      <div className="card bg-gradient-to-br from-accent-blue/20 to-accent-teal/20 border-accent-blue/30 mb-8">
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 bg-accent-blue/30 rounded-full flex items-center justify-center flex-shrink-0">
            <Brain className="w-8 h-8 text-accent-blue" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">
              🚧 Under Development
            </h2>
            <p className="text-gray-300">
              We're building powerful explainability features to help you understand model predictions. 
              Check back soon for attention heatmaps, SHAP values, and more!
            </p>
          </div>
        </div>
      </div>

      {/* Planned Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <div className="flex items-start space-x-3 mb-3">
            <div className="w-10 h-10 bg-accent-blue/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <Target className="w-5 h-5 text-accent-blue" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">
                Attention Visualization
              </h3>
              <p className="text-sm text-gray-400 mt-1">
                See which molecular substructures the GCN model focuses on when making predictions
              </p>
            </div>
          </div>
          <ul className="text-sm text-gray-400 space-y-1 ml-13">
            <li>• Interactive atom-level attention heatmaps</li>
            <li>• Bond importance highlighting</li>
            <li>• Layer-wise attention flow visualization</li>
          </ul>
        </div>

        <div className="card">
          <div className="flex items-start space-x-3 mb-3">
            <div className="w-10 h-10 bg-accent-teal/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <TrendingUp className="w-5 h-5 text-accent-teal" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">
                SHAP Values
              </h3>
              <p className="text-sm text-gray-400 mt-1">
                Quantify feature contributions using SHAP (SHapley Additive exPlanations)
              </p>
            </div>
          </div>
          <ul className="text-sm text-gray-400 space-y-1 ml-13">
            <li>• Feature importance rankings</li>
            <li>• Force plots for individual predictions</li>
            <li>• Summary plots across datasets</li>
          </ul>
        </div>

        <div className="card">
          <div className="flex items-start space-x-3 mb-3">
            <div className="w-10 h-10 bg-toxic-orange/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <Sparkles className="w-5 h-5 text-toxic-orange" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">
                Substructure Alerts
              </h3>
              <p className="text-sm text-gray-400 mt-1">
                Identify known toxicophores and structural alerts in your molecules
              </p>
            </div>
          </div>
          <ul className="text-sm text-gray-400 space-y-1 ml-13">
            <li>• ToxAlert pattern matching</li>
            <li>• Literature-backed toxicophores</li>
            <li>• Custom alert rule creation</li>
          </ul>
        </div>

        <div className="card">
          <div className="flex items-start space-x-3 mb-3">
            <div className="w-10 h-10 bg-safe-green/20 rounded-lg flex items-center justify-center flex-shrink-0">
              <Brain className="w-5 h-5 text-safe-green" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">
                Counterfactuals
              </h3>
              <p className="text-sm text-gray-400 mt-1">
                Discover minimal molecular modifications to change toxicity predictions
              </p>
            </div>
          </div>
          <ul className="text-sm text-gray-400 space-y-1 ml-13">
            <li>• Nearest neighbor analysis</li>
            <li>• Suggested modifications</li>
            <li>• Drug-likeness preservation</li>
          </ul>
        </div>
      </div>

      {/* Resources */}
      <div className="card mt-8 bg-dark-bg">
        <h3 className="text-lg font-semibold text-white mb-4">
          📚 Learn More About Model Interpretability
        </h3>
        <div className="space-y-2 text-sm text-gray-400">
          <p>
            • <a href="https://arxiv.org/abs/1706.03825" target="_blank" rel="noopener noreferrer" className="text-accent-blue hover:underline">
              Graph Attention Networks (Veličković et al., 2017)
            </a>
          </p>
          <p>
            • <a href="https://arxiv.org/abs/1705.07874" target="_blank" rel="noopener noreferrer" className="text-accent-blue hover:underline">
              A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)
            </a>
          </p>
          <p>
            • <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6364502/" target="_blank" rel="noopener noreferrer" className="text-accent-blue hover:underline">
              Structural Alerts for Toxicity Prediction (Enoch et al., 2019)
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default ExplainabilityPage;

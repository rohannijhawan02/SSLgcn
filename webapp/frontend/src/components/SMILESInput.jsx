import { CheckCircle, XCircle, Loader, RotateCcw } from 'lucide-react';

const SMILESInput = ({ smiles, setSMILES, validationResult, isValidating, onValidate, onReset }) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onValidate();
    }
  };

  // Example SMILES for quick testing
  const exampleSMILES = [
    { name: 'Ethanol', smiles: 'CCO' },
    { name: 'Benzene', smiles: 'c1ccccc1' },
    { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  ];

  return (
    <div className="space-y-4">
      {/* SMILES Input Field */}
      <div>
        <label htmlFor="smiles-input" className="block text-sm font-medium text-gray-300 mb-2">
          Enter SMILES String
        </label>
        <div className="relative">
          <textarea
            id="smiles-input"
            value={smiles}
            onChange={(e) => setSMILES(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="e.g., CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
            rows={3}
            className="input-field font-mono text-sm resize-none"
            disabled={isValidating}
          />

          {/* Validation Icon */}
          {validationResult && (
            <div className="absolute top-3 right-3">
              {validationResult.valid ? (
                <CheckCircle className="w-6 h-6 text-safe-green" />
              ) : (
                <XCircle className="w-6 h-6 text-toxic-red" />
              )}
            </div>
          )}
        </div>

        {/* Error Message */}
        {validationResult && !validationResult.valid && validationResult.error && (
          <p className="mt-2 text-sm text-toxic-red flex items-center">
            <XCircle className="w-4 h-4 mr-1" />
            {validationResult.error}
          </p>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-3">
        <button
          onClick={onValidate}
          disabled={!smiles.trim() || isValidating}
          className="btn-primary flex-1 flex items-center justify-center space-x-2"
        >
          {isValidating ? (
            <>
              <Loader className="w-4 h-4 animate-spin" />
              <span>Validating...</span>
            </>
          ) : (
            <>
              <CheckCircle className="w-4 h-4" />
              <span>Validate SMILES</span>
            </>
          )}
        </button>

        <button
          onClick={onReset}
          className="btn-secondary flex items-center space-x-2"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Reset</span>
        </button>
      </div>

      {/* Example SMILES */}
      <div>
        <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">
          Quick Examples
        </p>
        <div className="grid grid-cols-2 gap-2">
          {exampleSMILES.map((example) => (
            <button
              key={example.smiles}
              onClick={() => setSMILES(example.smiles)}
              className="text-left px-3 py-2 bg-dark-bg hover:bg-gray-800 rounded border border-dark-border 
                       transition-all hover:border-accent-blue group"
            >
              <p className="text-sm font-medium text-gray-300 group-hover:text-accent-blue">
                {example.name}
              </p>
              <p className="text-xs font-mono text-gray-500 truncate">
                {example.smiles}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* SMILES Guide */}
      <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-300 mb-2">
          💡 SMILES Format Tips
        </h4>
        <ul className="text-xs text-gray-400 space-y-1">
          <li>• Use standard SMILES notation (e.g., CCO for ethanol)</li>
          <li>• Aromatic rings use lowercase (c for aromatic carbon)</li>
          <li>• Branches use parentheses: CC(C)C</li>
          <li>• Double bonds: C=C, Triple bonds: C#C</li>
          <li>• Ring closures use numbers: C1CCCCC1 (cyclohexane)</li>
        </ul>
      </div>
    </div>
  );
};

export default SMILESInput;

import { CheckSquare, Square } from 'lucide-react';

const EndpointSelector = ({ endpoints, selectedEndpoints, setSelectedEndpoints, presets }) => {
  const handleToggleEndpoint = (endpointId) => {
    setSelectedEndpoints((prev) =>
      prev.includes(endpointId)
        ? prev.filter((id) => id !== endpointId)
        : [...prev, endpointId]
    );
  };

  const handleSelectAll = () => {
    setSelectedEndpoints(endpoints.map((e) => e.id));
  };

  const handleDeselectAll = () => {
    setSelectedEndpoints([]);
  };

  const handleApplyPreset = async (presetKey) => {
    // Preset-to-endpoint mappings based on backend configuration
    const presetMappings = {
      'all': endpoints.map((e) => e.id),
      'nuclear_receptor': endpoints.filter((e) => e.category === 'Nuclear Receptor').map((e) => e.id),
      'stress_response': endpoints.filter((e) => e.category === 'Stress Response').map((e) => e.id),
      'environmental': ['NR-AhR', 'NR-ER', 'NR-AR', 'SR-ARE'].filter(id => endpoints.find(e => e.id === id)),
      'endocrine': ['NR-AR', 'NR-ER', 'NR-Aromatase', 'NR-PPAR-gamma'].filter(id => endpoints.find(e => e.id === id))
    };
    
    const endpointIds = presetMappings[presetKey] || [];
    setSelectedEndpoints(endpointIds);
  };

  // Group endpoints by category
  const nuclearReceptor = endpoints.filter((e) => e.category === 'Nuclear Receptor');
  const stressResponse = endpoints.filter((e) => e.category === 'Stress Response');

  // Convert presets object to array for rendering
  const presetsArray = presets && typeof presets === 'object' 
    ? Object.entries(presets).map(([key, value]) => ({ key, ...value }))
    : [];

  return (
    <div className="space-y-4">
      {/* Preset Buttons */}
      {presetsArray.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">
            Quick Presets
          </p>
          <div className="flex flex-wrap gap-2">
            {presetsArray.map((preset) => (
              <button
                key={preset.key}
                onClick={() => handleApplyPreset(preset.key)}
                className="px-3 py-1.5 bg-dark-bg hover:bg-gray-800 rounded border border-dark-border 
                         text-xs font-medium text-gray-300 hover:text-accent-blue hover:border-accent-blue 
                         transition-all"
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Select/Deselect All */}
      <div className="flex space-x-2">
        <button
          onClick={handleSelectAll}
          className="btn-secondary flex-1 text-xs"
        >
          Select All ({endpoints.length})
        </button>
        <button
          onClick={handleDeselectAll}
          className="btn-secondary flex-1 text-xs"
        >
          Deselect All
        </button>
      </div>

      {/* Nuclear Receptor Endpoints */}
      {nuclearReceptor.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
            <span className="w-2 h-2 bg-toxic-orange rounded-full mr-2"></span>
            Nuclear Receptor ({nuclearReceptor.length})
          </h4>
          <div className="grid grid-cols-1 gap-2">
            {nuclearReceptor.map((endpoint) => (
              <label
                key={endpoint.id}
                className="flex items-start space-x-3 p-3 bg-dark-bg border border-dark-border rounded 
                         hover:bg-gray-800 cursor-pointer transition-all group"
              >
                <div className="flex-shrink-0 mt-0.5">
                  {selectedEndpoints.includes(endpoint.id) ? (
                    <CheckSquare className="w-5 h-5 text-accent-blue" />
                  ) : (
                    <Square className="w-5 h-5 text-gray-500 group-hover:text-gray-400" />
                  )}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-300 group-hover:text-white">
                    {endpoint.name}
                  </p>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {endpoint.description}
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={selectedEndpoints.includes(endpoint.id)}
                  onChange={() => handleToggleEndpoint(endpoint.id)}
                  className="sr-only"
                />
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Stress Response Endpoints */}
      {stressResponse.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
            <span className="w-2 h-2 bg-accent-teal rounded-full mr-2"></span>
            Stress Response ({stressResponse.length})
          </h4>
          <div className="grid grid-cols-1 gap-2">
            {stressResponse.map((endpoint) => (
              <label
                key={endpoint.id}
                className="flex items-start space-x-3 p-3 bg-dark-bg border border-dark-border rounded 
                         hover:bg-gray-800 cursor-pointer transition-all group"
              >
                <div className="flex-shrink-0 mt-0.5">
                  {selectedEndpoints.includes(endpoint.id) ? (
                    <CheckSquare className="w-5 h-5 text-accent-blue" />
                  ) : (
                    <Square className="w-5 h-5 text-gray-500 group-hover:text-gray-400" />
                  )}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-300 group-hover:text-white">
                    {endpoint.name}
                  </p>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {endpoint.description}
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={selectedEndpoints.includes(endpoint.id)}
                  onChange={() => handleToggleEndpoint(endpoint.id)}
                  className="sr-only"
                />
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Selection Summary */}
      <div className="bg-accent-blue/10 border border-accent-blue/30 rounded-lg p-3">
        <p className="text-sm text-accent-blue">
          <span className="font-semibold">{selectedEndpoints.length}</span> endpoint
          {selectedEndpoints.length !== 1 ? 's' : ''} selected
        </p>
      </div>
    </div>
  );
};

export default EndpointSelector;

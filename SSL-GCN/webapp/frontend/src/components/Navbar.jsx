import { Link, useLocation } from 'react-router-dom';
import { Beaker, Brain, FileText, Info } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Predict', icon: Beaker },
    { path: '/explainability', label: 'Explainability', icon: Brain },
    { path: '/research', label: 'Research', icon: FileText },
    { path: '/about', label: 'About', icon: Info },
  ];

  return (
    <nav className="bg-dark-card border-b border-dark-border sticky top-0 z-50 backdrop-blur-lg bg-opacity-95">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="p-2 bg-accent-blue rounded-lg group-hover:bg-blue-600 transition-colors">
              <Beaker className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-100">ToxPredict</h1>
              <p className="text-xs text-gray-400">Research-Grade ML Platform</p>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
                    isActive
                      ? 'bg-accent-blue text-white'
                      : 'text-gray-300 hover:bg-dark-bg hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Mobile menu button (for future implementation) */}
          <div className="md:hidden">
            <button className="p-2 rounded-lg text-gray-300 hover:bg-dark-bg">
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

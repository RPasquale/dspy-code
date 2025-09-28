import React from 'react';
import { api } from '../api/client';

const KEY = 'dspy_profile';

const ProfileSwitcher: React.FC = () => {
  const [profile, setProfile] = React.useState<string>(() => (localStorage.getItem(KEY) || 'balanced'));
  const [isOpen, setIsOpen] = React.useState(false);
  
  // Load profile from backend once
  React.useEffect(() => {
    let mounted = true;
    api.getProfile().then((resp) => {
      const p = (resp?.profile || '').trim();
      if (mounted && p) {
        setProfile(p);
        try { localStorage.setItem(KEY, p); } catch {}
      }
    }).catch(() => {});
    return () => { mounted = false; };
  }, []);
  
  // Persist locally + send to backend
  React.useEffect(() => {
    try { localStorage.setItem(KEY, profile); } catch {}
    api.updateConfig({ type: 'profile', value: profile }).catch(() => {});
  }, [profile]);

  const profiles = [
    { value: 'fast', label: 'Fast', description: 'Quick responses', icon: '⚡' },
    { value: 'balanced', label: 'Balanced', description: 'Balanced performance', icon: '⚖️' },
    { value: 'maxquality', label: 'Max Quality', description: 'Highest quality', icon: '🎯' }
  ];

  const currentProfile = profiles.find(p => p.value === profile);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 hover:border-slate-300 transition-all duration-200 text-sm"
      >
        <span className="text-lg">{currentProfile?.icon || '⚖️'}</span>
        <span className="font-medium text-slate-700">
          {currentProfile?.label || profile}
        </span>
        <svg 
          className={`w-4 h-4 text-slate-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-56 bg-white rounded-lg border border-slate-200 shadow-lg z-50">
          <div className="p-2">
            {profiles.map((p) => (
              <button
                key={p.value}
                onClick={() => {
                  setProfile(p.value);
                  setIsOpen(false);
                }}
                className={`w-full text-left px-3 py-2 rounded-lg transition-all duration-200 ${
                  profile === p.value
                    ? 'bg-blue-50 text-blue-700 border border-blue-200'
                    : 'text-slate-700 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{p.icon}</span>
                  <div>
                    <div className="font-medium text-sm">{p.label}</div>
                    <div className="text-xs text-slate-500">{p.description}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfileSwitcher;
import { ReactNode } from 'react';

interface CardProps {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  variant?: 'default' | 'elevated' | 'outlined';
  className?: string;
}

const Card = ({ 
  title, 
  subtitle, 
  actions, 
  children, 
  variant = 'default',
  className = ''
}: CardProps) => {
  const baseClasses = 'card';
  
  const variantClasses = {
    default: 'cyber-card border border-cyan-500 shadow-sm neon-glow',
    elevated: 'cyber-card border border-cyan-500 shadow-lg neon-glow',
    outlined: 'cyber-card border-2 border-cyan-500 shadow-none neon-glow'
  };

  return (
    <div className={`${baseClasses} ${variantClasses[variant]} ${className}`}>
      {(title || subtitle || actions) && (
        <div className="card-header">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              {title && (
                <h3 className="text-lg font-semibold text-green-400 mb-1">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="text-sm text-green-200">
                  {subtitle}
                </p>
              )}
            </div>
            {actions && (
              <div className="flex items-center space-x-2 ml-4">
                {actions}
              </div>
            )}
          </div>
        </div>
      )}
      <div className="card-body">
        {children}
      </div>
    </div>
  );
};

export default Card;
import { ReactNode } from 'react';

interface CardProps {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  variant?: 'default' | 'elevated' | 'outlined' | 'interactive';
  className?: string;
  loading?: boolean;
  error?: string;
}

const Card = ({ 
  title, 
  subtitle, 
  actions, 
  children, 
  variant = 'default',
  className = '',
  loading = false,
  error
}: CardProps) => {
  const baseClasses = 'card';
  
  const variantClasses = {
    default: '',
    elevated: 'card-elevated',
    outlined: 'border-2',
    interactive: 'card-interactive'
  };

  return (
    <div className={`${baseClasses} ${variantClasses[variant]} ${className}`}>
      {(title || subtitle || actions) && (
        <div className="p-6 pb-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              {title && (
                <h3 className="text-lg font-semibold text-primary mb-1">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="text-sm text-secondary">
                  {subtitle}
                </p>
              )}
            </div>
            {actions && (
              <div className="flex items-center gap-2 ml-4">
                {actions}
              </div>
            )}
          </div>
        </div>
      )}
      <div className="p-6">
        {loading ? (
          <div className="space-y-3">
            <div className="skeleton h-4 w-3/4 rounded"></div>
            <div className="skeleton h-4 w-1/2 rounded"></div>
            <div className="skeleton h-4 w-5/6 rounded"></div>
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-error">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span className="text-sm">{error}</span>
          </div>
        ) : (
          children
        )}
      </div>
    </div>
  );
};

export default Card;
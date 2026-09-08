import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'cyan' | 'emerald' | 'amber' | 'rose' | 'purple' | 'blue' | 'neutral';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  dot?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'cyan',
  size = 'md',
  className = '',
  dot = false,
}) => {
  const variantStyles = {
    cyan: 'bg-cyan-950/70 text-cyan-400 border-cyan-800/80',
    emerald: 'bg-emerald-950/70 text-emerald-400 border-emerald-800/80',
    amber: 'bg-amber-950/70 text-amber-400 border-amber-800/80',
    rose: 'bg-rose-950/70 text-rose-400 border-rose-800/80',
    purple: 'bg-purple-950/70 text-purple-400 border-purple-800/80',
    blue: 'bg-blue-950/70 text-blue-400 border-blue-800/80',
    neutral: 'bg-slate-900/80 text-slate-300 border-slate-700/80',
  };

  const dotColors = {
    cyan: 'bg-cyan-400',
    emerald: 'bg-emerald-400',
    amber: 'bg-amber-400',
    rose: 'bg-rose-400',
    purple: 'bg-purple-400',
    blue: 'bg-blue-400',
    neutral: 'bg-slate-400',
  };

  const sizeStyles = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs font-medium',
    lg: 'px-3 py-1.5 text-sm font-medium',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border shadow-sm ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
    >
      {dot && <span className={`h-1.5 w-1.5 rounded-full ${dotColors[variant]} animate-pulse`} />}
      {children}
    </span>
  );
};

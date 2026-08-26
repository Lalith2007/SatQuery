import React from 'react';
import { SystemStatusView } from '../components/status/SystemStatusView';

export const SystemPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <SystemStatusView />
    </div>
  );
};

import type { FunctionalComponent } from 'preact';

interface DashboardProps {
    proxyHealth: any;
    serverHealth: any;
    webUiHealth: any;
    audioStatus: string;
}

export const Dashboard: FunctionalComponent<DashboardProps> = ({
    proxyHealth,
    serverHealth,
    webUiHealth,
    audioStatus,
}) => {
    const renderStatus = (health: any) => {
        if (!health) return <span style="color: #666">Checking...</span>;
        if (health.ok || health.status === 'ok') return <span style="color: #2e7d32">Healthy</span>;
        return <span style="color: #d32f2f">Unhealthy</span>;
    };

    return (
        <div class="card">
            <h3>Service Status</h3>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div>
                    <strong>Proxy:</strong> {renderStatus(proxyHealth)}
                </div>
                <div>
                    <strong>Server:</strong> {renderStatus(serverHealth)}
                </div>
                <div>
                    <strong>Web UI:</strong> {renderStatus(webUiHealth)}
                </div>
                <div>
                    <strong>Audio:</strong> <span class="badge">{audioStatus}</span>
                </div>
            </div>
            {proxyHealth && (
                <div style="margin-top: 10px; font-size: 12px; color: #666;">
                    Active Sessions: {proxyHealth.active_sessions} / {proxyHealth.max_sessions}
                </div>
            )}
        </div>
    );
};

import { useState, useRef, useEffect } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd';

export interface Widget {
  id: string;
  type: string;
  title: string;
  size: 'small' | 'medium' | 'large' | 'full';
  position: { x: number; y: number };
  data?: any;
  config?: Record<string, any>;
}

export interface DashboardWidgetProps {
  widgets: Widget[];
  onWidgetMove: (widgetId: string, newPosition: { x: number; y: number }) => void;
  onWidgetResize: (widgetId: string, newSize: 'small' | 'medium' | 'large' | 'full') => void;
  onWidgetRemove: (widgetId: string) => void;
  onWidgetEdit: (widgetId: string, config: Record<string, any>) => void;
  className?: string;
  editable?: boolean;
}

const DashboardWidget = ({
  widgets,
  onWidgetMove,
  onWidgetResize,
  onWidgetRemove,
  onWidgetEdit,
  className = '',
  editable = true
}: DashboardWidgetProps) => {
  const [draggedWidget, setDraggedWidget] = useState<string | null>(null);
  const [resizingWidget, setResizingWidget] = useState<string | null>(null);

  const handleDragStart = (result: any) => {
    setDraggedWidget(result.draggableId);
  };

  const handleDragEnd = (result: DropResult) => {
    setDraggedWidget(null);
    
    if (!result.destination) return;

    const widgetId = result.draggableId;
    const newIndex = result.destination.index;
    
    // Calculate new position based on grid
    const cols = 4; // 4-column grid
    const newPosition = {
      x: newIndex % cols,
      y: Math.floor(newIndex / cols)
    };

    onWidgetMove(widgetId, newPosition);
  };

  const getSizeClasses = (size: string) => {
    switch (size) {
      case 'small':
        return 'col-span-1 row-span-1';
      case 'medium':
        return 'col-span-2 row-span-1';
      case 'large':
        return 'col-span-2 row-span-2';
      case 'full':
        return 'col-span-4 row-span-1';
      default:
        return 'col-span-1 row-span-1';
    }
  };

  const renderWidget = (widget: Widget, index: number) => {
    const isDragging = draggedWidget === widget.id;
    const isResizing = resizingWidget === widget.id;

    return (
      <Draggable
        key={widget.id}
        draggableId={widget.id}
        index={index}
        isDragDisabled={!editable}
      >
        {(provided, snapshot) => (
          <div
            ref={provided.innerRef}
            {...provided.draggableProps}
            className={`
              relative bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm
              ${getSizeClasses(widget.size)}
              ${isDragging ? 'opacity-50' : ''}
              ${snapshot.isDragging ? 'shadow-lg' : ''}
              transition-all duration-200
            `}
          >
            {/* Widget Header */}
            <div
              {...provided.dragHandleProps}
              className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {widget.title}
              </h3>
              
              {editable && (
                <div className="flex items-center gap-2">
                  {/* Size Controls */}
                  <div className="flex items-center gap-1">
                    {['small', 'medium', 'large', 'full'].map(size => (
                      <button
                        key={size}
                        onClick={() => onWidgetResize(widget.id, size as any)}
                        className={`w-2 h-2 rounded-full ${
                          widget.size === size 
                            ? 'bg-blue-500' 
                            : 'bg-gray-300 hover:bg-gray-400'
                        }`}
                        title={`Resize to ${size}`}
                      />
                    ))}
                  </div>

                  {/* Widget Actions */}
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => onWidgetEdit(widget.id, widget.config || {})}
                      className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                      title="Edit widget"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                    </button>
                    
                    <button
                      onClick={() => onWidgetRemove(widget.id)}
                      className="p-1 text-gray-400 hover:text-red-600 dark:hover:text-red-400"
                      title="Remove widget"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Widget Content */}
            <div className="p-4">
              {renderWidgetContent(widget)}
            </div>

            {/* Resize Handle */}
            {editable && (
              <div
                className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize"
                onMouseDown={() => setResizingWidget(widget.id)}
              >
                <svg className="w-4 h-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M3 3h14v14H3V3zm2 2v10h10V5H5z"/>
                </svg>
              </div>
            )}
          </div>
        )}
      </Draggable>
    );
  };

  const renderWidgetContent = (widget: Widget) => {
    switch (widget.type) {
      case 'chart':
        return (
          <div className="h-48 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <p className="text-sm">Chart Widget</p>
            </div>
          </div>
        );

      case 'table':
        return (
          <div className="space-y-2">
            <div className="h-32 flex items-center justify-center text-gray-500">
              <div className="text-center">
                <svg className="w-8 h-8 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0V4a1 1 0 011-1h16a1 1 0 011 1v16a1 1 0 01-1 1H4a1 1 0 01-1-1z" />
                </svg>
                <p className="text-sm">Table Widget</p>
              </div>
            </div>
          </div>
        );

      case 'metric':
        return (
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
              {widget.data?.value || '0'}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {widget.data?.label || 'Metric'}
            </div>
            {widget.data?.change && (
              <div className={`text-xs mt-1 ${
                widget.data.change > 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {widget.data.change > 0 ? '+' : ''}{widget.data.change}%
              </div>
            )}
          </div>
        );

      case 'text':
        return (
          <div className="prose dark:prose-invert max-w-none">
            <p className="text-gray-600 dark:text-gray-300">
              {widget.data?.content || 'Text widget content goes here...'}
            </p>
          </div>
        );

      default:
        return (
          <div className="h-32 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <p className="text-sm">Unknown widget type: {widget.type}</p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className={`grid grid-cols-4 gap-4 ${className}`}>
      <DragDropContext onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
        <Droppable droppableId="dashboard" direction="horizontal">
          {(provided) => (
            <div
              ref={provided.innerRef}
              {...provided.droppableProps}
              className="contents"
            >
              {widgets.map((widget, index) => renderWidget(widget, index))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
    </div>
  );
};

export default DashboardWidget;

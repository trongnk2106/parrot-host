def update_task_info(
        result=None,
        vector_result=None
):
    try:
        if vector_result:
            response = f"""
                update task_management 
                set vector_result = {':vector_result' if vector_result else "'null'"},
                 result = {':result' if result else "'null'"}, status = :status
                where task_id = :task_id
                returning id;
            """
        else:
            response = f"""
                            update task_management 
                            set result = {':result' if result else "'null'"}, status = :status
                            where task_id = :task_id
                            returning id;
                        """
        return response
    except Exception as e:
        # TODO: Not implement logger
        return None
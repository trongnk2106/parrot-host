UPDATE_RESULT_FOR_TASK_QUERY = """update task_management set status=:status,result=:result where task_id=:task_id returning id;"""


def select_vector_info():
    try:
        response = f"""
            select vector_result from task_management 
            where task_id = {':task_id'};
        """
        return response
    except Exception as e:
        return None